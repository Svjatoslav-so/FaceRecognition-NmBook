import json
import multiprocessing
import threading
import time
from pathlib import Path

import sqlalchemy as sa
import sqlalchemy.ext.compiler
from deepface.commons import distance as dst
from sqlalchemy import create_engine, Column, Integer, Text, Float, String, ForeignKey, Table, Engine, event, Index, \
    UniqueConstraint, select, Boolean, text
from sqlalchemy.orm import relationship, declarative_base, sessionmaker, Session, aliased
from sqlalchemy.schema import DDLElement
from sqlalchemy.sql import table
from tqdm import tqdm

import tool_module
from web.resultDBManager import ResultDBManager

MIN_FACE_AREA = 650


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """
        В SQLite3 по умолчанию отключена поддержка внешних ключей,
        команда PRAGMA позволяет включить внешние ключи в базах данных SQLite.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


Base = declarative_base()


class Face(Base):
    __tablename__ = 'faces'
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False, unique=True)
    embedding = Column(Text, nullable=False)
    x1 = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    x2 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)
    is_side_view = Column(Boolean, nullable=False, default=False)
    photo_id = Column(String(200), ForeignKey('photos.id', ondelete='CASCADE'))
    photo = relationship("Photo", back_populates='faces')

    def __repr__(self):
        return f"<Face(id:{self.id}, embedding:{self.embedding}, " \
               f"x1: {self.x1}, y1: {self.y1}, x2: {self.x2}, y2: {self.y2}, " \
               f"is_side_view: {self.is_side_view}, " \
               f"photo_id: {self.photo_id})>"

    def to_dict(self):
        return {'id': self.id, 'embedding': self.embedding, 'x1': self.x1, 'y1': self.y1, 'x2': self.x2, 'y2': self.y2,
                'is_side_view': self.is_side_view, 'photo_id': self.photo_id}

    def to_dict_with_relationship(self):
        dic = self.to_dict()
        dic['photo'] = self.photo.to_dict()
        return dic


photo_document = Table('photo_document', Base.metadata,
                       Column('photo_id', String(200), ForeignKey('photos.id'), primary_key=True),
                       Column('doc_id', Integer(), ForeignKey('doc.id'), primary_key=True))


class Photo(Base):
    __tablename__ = 'photos'
    id = Column(String(200), primary_key=True, nullable=False, unique=True)
    title = Column(Text)
    faces = relationship("Face", back_populates='photo')
    docs = relationship("Document", secondary=photo_document, back_populates="photos")

    def __repr__(self):
        return f"<Photo(id:{self.id}, title:{self.title}, doc_id: {self.docs})>"

    def to_dict(self):
        return {'id': self.id, 'title': self.title}

    def to_dict_with_relationship(self, face=True, doc=True):
        dic = self.to_dict()
        if face:
            dic['faces'] = [f.to_dict() for f in self.faces]
        if doc:
            dic['docs'] = [d.to_dict() for d in self.docs]
        return dic


class Document(Base):
    __tablename__ = 'doc'
    id = Column(Integer, primary_key=True, nullable=False, unique=True)
    photos = relationship("Photo", secondary=photo_document, back_populates="docs")

    def __repr__(self):
        return f"<Document(id:{self.id})>"

    def to_dict(self):
        return {'id': self.id}

    def to_dict_with_relationship(self):
        return {'id': self.id, 'photos': [p.to_dict() for p in self.photos]}


class Distance(Base):
    __tablename__ = 'distances'
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False, unique=True)
    face1_id = Column(Integer, ForeignKey('faces.id'), nullable=False)
    face2_id = Column(Integer, ForeignKey('faces.id'), nullable=False)
    distance = Column(Float, nullable=False)
    face1 = relationship("Face", primaryjoin="Distance.face1_id == Face.id")
    face2 = relationship("Face", primaryjoin="Distance.face2_id == Face.id")
    idx_face1_face2_distance = Index('idx_face1_face2_distance', face1_id, face2_id, distance)

    __table_args__ = (
        UniqueConstraint("face1_id", "face2_id"),
    )

    def __repr__(self):
        return f"<Distance(id:{self.id}, face_1:{self.face1_id}, face_2: {self.face2_id}, distance: {self.distance})>"


# ---------Для работы с представлениями (View)---------------
# В качестве примера использовался: https://github.com/sqlalchemy/sqlalchemy/wiki/Views
# Альтернативы: https://stackoverflow.com/questions/9766940/how-to-create-an-sql-view-with-sqlalchemy
class CreateView(DDLElement):
    def __init__(self, name, selectable):
        self.name = name
        self.selectable = selectable


class DropView(DDLElement):
    def __init__(self, name):
        self.name = name


@sa.ext.compiler.compiles(CreateView)
def _create_view(element, compiler, **kw):
    return "CREATE VIEW %s AS %s" % (
        element.name,
        compiler.sql_compiler.process(element.selectable, literal_binds=True),
    )


@sa.ext.compiler.compiles(DropView)
def _drop_view(element, compiler, **kw):
    return "DROP VIEW %s" % (element.name)


def view_exists(ddl, target, connection, **kw):
    return ddl.name in sa.inspect(connection).get_view_names()


def view_doesnt_exist(ddl, target, connection, **kw):
    return not view_exists(ddl, target, connection, **kw)


def view(name, metadata, selectable):
    t = table(name)
    t._columns._populate_separate_keys(
        col._make_proxy(t) for col in selectable.selected_columns
    )
    sa.event.listen(
        metadata,
        "after_create",
        CreateView(name, selectable).execute_if(callable_=view_doesnt_exist),
    )
    sa.event.listen(
        metadata, "before_drop", DropView(name).execute_if(callable_=view_exists)
    )
    return t


# ---------------------------------------------------------------


class DBManager:
    def __init__(self, db_name: str, echo=False):
        self.db_name = db_name
        self.engine = create_engine(f"sqlite:///{db_name}", echo=echo)

        # создаем представление face_view
        self.face_view = view(
            name='face_view',
            metadata=Base.metadata,
            selectable=select(Face).filter(
                (Face.x2 - Face.x1) * (Face.y2 - Face.y1) > MIN_FACE_AREA,
                Face.is_side_view == False
            )
        )
        # Делаем класс FaceView для работы с представлением face_view через ORM
        self.FaceView = type('FaceView', (Base,), {'__table__': self.face_view, })
        self.FaceView.__repr__ = Face.__repr__
        self.FaceView.to_dict = Face.to_dict
        self.FaceView.to_dict_with_relationship = Face.to_dict_with_relationship

        # создаем представление distance_view
        # face_table = aliased(self.FaceView, name='face_table')
        self.distance_view = view(
            name='distance_view',
            metadata=Base.metadata,
            selectable=select(Distance).join(self.FaceView, self.FaceView.id == Distance.face1_id).
            join(photo_document, self.FaceView.photo_id == photo_document.c.photo_id).
            filter(
                Distance.face2_id.in_(select(self.FaceView.id)),
                # Distance.face1_id.in_(select(self.FaceView.id)),  # ?
                Distance.face2_id < Distance.face1_id,
                photo_document.c.doc_id.notin_(
                    select(photo_document.c.doc_id).
                    join(self.FaceView, self.FaceView.photo_id == photo_document.c.photo_id).
                    filter(self.FaceView.id == Distance.face2_id))
            )
        )
        # Делаем класс DistanceView для работы с представлением distance_view через ORM
        self.DistanceView = type('DistanceView', (Base,), {'__table__': self.distance_view, })
        self.DistanceView.__repr__ = Distance.__repr__

        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        # тест
        # with self.Session() as session:
        #     print(session.query(self.FaceView).all())

    # def __del__(self):
    #     """
    #         Удаляет представления self.face_view и self.distance_view при удалении объекта DBManager.
    #         (Использовался при отладке, так как изменения внесенные в структуру представления не отражаются в бд,
    #         если это представление уже существует в бд)
    #     """
    #     print('DBManager object deleted successfully')
    #     with self.Session() as session:
    #         session.execute(text(f'DROP VIEW {self.face_view.name}'))
    #         session.execute(text(f'DROP VIEW {self.distance_view.name}'))

    def fill_doc_photos_faces_from_file(self, embedding_file, metadata_file):
        """
            Заполняет базу данных данными из json файлов:
            embedding_file - файл с кодировками (который генерит deep_face_module_v2.all_df_encodings_to_file)
            metadata_file - файл с метаданными фото
        """
        with self.Session() as session:
            with Path(embedding_file).open(encoding="utf8") as ef, Path(metadata_file).open(encoding="utf8") as mf:
                try:
                    data = json.load(ef)
                except json.decoder.JSONDecodeError:
                    raise ValueError(f'Не корректное содержимое файла {embedding_file}. Ожидается JSON.')
                try:
                    metadata = json.load(mf)['by_photo']
                except json.decoder.JSONDecodeError:
                    raise ValueError(f'Не корректное содержимое файла {metadata}. Ожидается JSON.')

                for photo in data:
                    photo_obj = Photo(id=self._get_filename_from_path(photo['path']).split('.')[0],
                                      title=metadata[self._get_filename_from_path(photo['path'])]['title'])
                    for face in photo['faces']:
                        face_obj = Face(embedding=json.dumps(face['encoding']),
                                        x1=face['face_area'][0],
                                        y1=face['face_area'][1],
                                        x2=face['face_area'][2],
                                        y2=face['face_area'][3],
                                        is_side_view=face['is_side_view'])
                        session.add(face_obj, )
                        session.flush()
                        photo_obj.faces.append(face_obj)
                    session.add(photo_obj)

                    for doc_id in metadata[self._get_filename_from_path(photo['path'])]['docs']:
                        doc_id = doc_id.get('key', doc_id)
                        doc_obj = session.query(Document).get(doc_id)
                        if not doc_obj:
                            doc_obj = Document(id=doc_id)
                            session.add(doc_obj)
                        session.flush()
                        photo_obj.docs.append(doc_obj)
                session.commit()

    @staticmethod
    def _calculating_process(e_dict, cpd_range, d_metric, lock, disable, db_name, commit_block_size):
        """
            Выполняется отдельным процессом
            e_dict - embeddings_dict,
            cpd_range - диапазон лиц из cpd_range, для которых рассчитываются расстояния этим процессом,
            d_metric - distance_metric,
            lock - блокировка на доступ к базе,
            disable - если True, то прогресс-бар отображаться не будет,
            db_name - имя бд к которой нужно подключиться,
            commit_block_size - размер блока записей, по достижении которого будет происходить коммит в бд.

        """
        engine = create_engine(f"sqlite:///{db_name}")
        with Session(engine) as session:
            result = []
            for face1_id, face1_embedding in tqdm(list(e_dict.items())[cpd_range[0]: cpd_range[1]],
                                                  desc='_calculating_process', disable=disable):
                for face2_id, face2_embedding in e_dict.items():
                    if d_metric == 'cosine':
                        distance = dst.findCosineDistance(face1_embedding,
                                                          face2_embedding)
                    elif d_metric == 'euclidean':
                        distance = dst.findEuclideanDistance(face1_embedding,
                                                             face2_embedding)
                    elif d_metric == 'euclidean_l2':
                        distance = dst.findEuclideanDistance(dst.l2_normalize(face1_embedding),
                                                             dst.l2_normalize(face2_embedding))
                    else:
                        raise ValueError("Invalid distance_metric passed - ", d_metric)

                    result.append(Distance(face1_id=face1_id, face2_id=face2_id, distance=distance))
                # будем добавлять партиями по 10000 записей (если небольшие порции - долго, если большие - памяти много)
                if len(result) > commit_block_size:
                    lock.acquire()
                    try:
                        session.add_all(result)
                        session.commit()
                        result = []
                        # print("Commit")
                    finally:
                        lock.release()
            # если еще остались не добавленные записи
            if len(result) > 0:
                lock.acquire()
                try:
                    session.add_all(result)
                    session.commit()
                finally:
                    lock.release()

    def calculate_distance_matrix(self, distance_metric="cosine", disable=False, process_count=None,
                                  commit_block_size=100000):
        """
            Заполняет таблицу Distance значениями расстояний для всех лиц из Face
            Получается n^2 записей, где n - количество лиц в Face.
            distance_metric - может быть cosine, euclidean, euclidean_l2
            disable - если True, то прогресс-бар отображаться не будет
            process_count - количество процессов на которых будет выполняться расчет и добавление в бд
            commit_block_size - размер блока записей, по достижении которого будет происходить коммит в бд,
                                чем больше блок тем быстрее выполнится, но больше оперативной памяти потребуется,
                                чем меньше размер блока тем медленнее, но и меньший объем памяти потребуется.
        """
        with self.Session() as session:
            all_faces = session.query(Face).all()
        embeddings_dict = {}
        for f in all_faces:
            embeddings_dict[f.id] = json.loads(f.embedding)
        process_list = []
        lock = multiprocessing.Lock()
        process_count, pt_count = tool_module.get_optimal_process_count(len(embeddings_dict), process_count)
        try:
            for i in range(process_count):
                process_list.append(multiprocessing.Process(target=self._calculating_process,
                                                            kwargs={
                                                                'e_dict': embeddings_dict,
                                                                'cpd_range': (i * pt_count, i * pt_count + pt_count),
                                                                'd_metric': distance_metric,
                                                                'lock': lock,
                                                                'disable': disable,
                                                                'db_name': self.db_name,
                                                                'commit_block_size': commit_block_size
                                                            }))
                process_list[i].start()
            for p in process_list:
                p.join()
        except KeyboardInterrupt as ki:
            for p in process_list:
                p.kill()
            raise ki
        print("All process finished!")

    @staticmethod
    def _get_filename_from_path(path):
        """ Принимает путь к файлу (path: str) возвращает имя файла (str) """
        path = Path(path)
        return path.name

    def drop_all_tables(self):
        """ Удаляет все таблицы """
        Base.metadata.drop_all(self.engine)

    def delete_all_data_in_table(self, table_class):
        """
            Удаляет все записи в таблице table_class. Где table_class может быть:
                Face,
                Photo,
                Document,
                Distance
        """
        if table_class in [Face, Photo, Document, Distance]:
            with self.Session() as session:
                session.query(table_class).delete()
                session.commit()
        else:
            raise ValueError(f'table_class - должен быть один из классов [Face, Photo, Document, Distance],'
                             f' а передан {table_class} типа {type(table_class)}')

    def get_photo_group_list_of_similar(self, threshold, min_face_area=MIN_FACE_AREA):
        """ Получаем список групп
                threshold - пороговое значение для расстояния между векторами лиц
                min_face_area - пороговое значение для площади лица(используется если запрос строится без self.FaceView)
            Возвращает список словарей вида:
            {
             'photo': {'id': str, 'title': str, 'docs': [int]},
             'face':
                {'id': int, 'embedding': [float], 'x1': float, 'y1': float, 'x2': float, 'y2': float, 'photo_id': str}
            }

        """
        with self.Session() as session:
            # # Вариант 1 (без представлений)
            # face_table = aliased(Face, name='face_table')
            # photos_faces = session.query(Photo, Face).join(Face).filter(
            #     (Face.x2 - Face.x1) * (Face.y2 - Face.y1) > min_face_area,
            #     Face.is_side_view == False
            # ).join(photo_document, Face.photo_id == photo_document.c.photo_id). \
            #     filter(Face.id == Distance.face1_id,
            #            Distance.distance < threshold,
            #            Distance.face2_id < Distance.face1_id,
            #            photo_document.c.doc_id.notin_(
            #                select(photo_document.c.doc_id).join(Face,
            #                                                     Face.photo_id == photo_document.c.photo_id).filter(
            #                    Face.id == Distance.face2_id))
            #            ).join(face_table, face_table.id == Distance.face2_id).filter(
            #     (face_table.x2 - face_table.x1) * (face_table.y2 - face_table.y1) > min_face_area,
            #     face_table.is_side_view == False
            # ).distinct().order_by(Photo.title).all()  # .order_by(Photo.title)

            # Вариант 2 (c двумя представлениями для Face - self.FaceView и для Distance - self.DistanceView)
            photos_faces = session.query(Photo, self.FaceView).join(self.FaceView). \
                join(self.DistanceView, self.FaceView.id == self.DistanceView.face1_id, ). \
                filter(self.DistanceView.distance < threshold) \
                .distinct().order_by(Photo.title).all()  # .order_by(Photo.title)

            return [{'photo': p.to_dict_with_relationship(face=False), 'face': f.to_dict()} for p, f in photos_faces]

    @staticmethod
    def _get_bookmarked(db: str, origin_photo_id: str, origin_face_x1: float, origin_face_y1: float,
                        origin_face_x2: float, origin_face_y2: float, result_list: list = None):
        """
            Используется в get_group_photo_with_face для получения списка закладок для origin_photo

            db: str - расположение базы данных с закладками,
            origin_photo_id: str - id origin_photo фото,
            origin_face_x1: float - координата лица с origin_photo,
            origin_face_y1: float - координата лица с origin_photo,
            origin_face_x2: float - координата лица с origin_photo,
            origin_face_y2: float - координата лица с origin_photo,
            result_list: list = None - контейнер для результатов

            Может быть использован в отдельном потоке, тогда результат будет помещен в result_list (если он передан)
            Возвращает список словарей описывающих фото из закладок, похожие на origin_photo.
            Структура такая:
            [{
                'id': id-лица, в базе данных закладок!!!,
                'photo_id': id-фото ,
                'x1': x1 - координата лица,
                'y1': y1 - координата лица,
                'x2': x2 - координата лица,
                'y2': y2 - координата лица,
                'photo_title': подпись к фото,
                'docs': строка с номерами документов к которым относится фото
            }]
        """
        result_manager = ResultDBManager(db)
        similar = result_manager.get_all_similar_of(origin_photo_id, origin_face_x1, origin_face_y1, origin_face_x2,
                                                    origin_face_y2)
        if not (result_list is None):
            result_list.extend(similar)
        return similar

    def get_group_photo_with_face(self, origin_face, threshold, min_face_area=MIN_FACE_AREA, bookmarks_db=None):
        """
            Получаем фото группы
                origin_face - id лица, которое образовало группу
                threshold - пороговое значение для расстояния между векторами лиц
                min_face_area - пороговое значение для площади лица(используется если запрос строится без self.FaceView)
            Возвращает список словарей вида:
            {
             'photo': {'id': str, 'title': str, 'docs': [int]},
             'face':
                {'id': int, 'embedding': [float], 'x1': float, 'y1': float, 'x2': float, 'y2': float, 'photo_id': str},
             'is_bookmarked': bool
            }
         """
        with self.Session() as session:
            origin_face = session.query(Face).get(origin_face)
            bookmarked_similar = list()
            t1 = None
            if bookmarks_db:
                t1 = threading.Thread(target=self._get_bookmarked, kwargs={'db': bookmarks_db,
                                                                           'origin_photo_id': origin_face.photo_id,
                                                                           'origin_face_x1': origin_face.x1,
                                                                           'origin_face_y1': origin_face.y1,
                                                                           'origin_face_x2': origin_face.x2,
                                                                           'origin_face_y2': origin_face.y2,
                                                                           'result_list': bookmarked_similar})
                t1.start()

            # # Вариант 1 (без представлений)
            # # origin_photo = session.query(Photo).get(origin_photo)
            # photos = session.query(Photo, Face).join(Face). \
            #     filter((Face.x2 - Face.x1) * (Face.y2 - Face.y1) > min_face_area
            #            , Face.is_side_view == False
            #            ). \
            #     join(photo_document).filter(Face.id == Distance.face2_id,
            #                                 # Distance.face1_id.in_([f.id for f in origin_photo.faces]),
            #                                 Distance.face1_id == origin_face.id,
            #                                 Distance.distance < threshold,
            #                                 Distance.face2_id < Distance.face1_id,  # ?
            #                                 Distance.face1_id != Distance.face2_id,
            #                                 photo_document.c.doc_id.notin_([d.id for d in origin_face.photo.docs])). \
            #     distinct().group_by(Photo.id).all()  # group_by(Photo.id)

            # Вариант 2 (c представлением для Face - self.FaceView)
            photos = session.query(Photo, self.FaceView).join(self.FaceView). \
                join(photo_document).filter(self.FaceView.id == Distance.face2_id,
                                            # Distance.face1_id.in_([f.id for f in origin_photo.faces]),
                                            Distance.face1_id == origin_face.id,
                                            Distance.distance < threshold,
                                            Distance.face2_id < Distance.face1_id,  # ?
                                            photo_document.c.doc_id.notin_([d.id for d in origin_face.photo.docs])). \
                distinct().group_by(Photo.id).all()  # group_by(Photo.id)

            # # Вариант 3 (c двумя представлениями для Face - self.FaceView и для Distance - self.DistanceView)
            # photos = session.query(Photo, self.FaceView).join(self.FaceView). \
            #     join(self.DistanceView, self.FaceView.id == self.DistanceView.face2_id).filter(
            #     self.DistanceView.face1_id == origin_face.id,
            #     self.DistanceView.distance < threshold,
            # ). \
            #     distinct().group_by(Photo.id).all()  # group_by(Photo.id)
            if t1:
                t1.join()
            print('BOOKMARKED_SIMILAR: ', bookmarked_similar)

            group = []
            for p, f in photos:
                photo_with_face = {'photo': p.to_dict_with_relationship(face=False), 'face': f.to_dict(),
                                   'is_bookmarked': False}
                for b in bookmarked_similar:
                    if b['photo_id'] == p.id \
                            and b['x1'] == f.x1 \
                            and b['y1'] == f.y1 \
                            and b['x2'] == f.x2 \
                            and b['y2'] == f.y2:
                        photo_with_face['is_bookmarked'] = True
                        break
                group.append(photo_with_face)
            return group


if __name__ == "__main__":
    print("START")
    start_time = time.time()

    directory = 'D:/Hobby/NmProject/nmbook_photo/web/static/out'
    manager = DBManager('web/db/NmBookPhoto(arcface-retinaface)-profile.db', echo=True)
    # manager.fill_doc_photos_faces_from_file(directory + "/photo/dfv2_arcface_encodings_with_profile.json",
    #                                         directory + "/metadata.json")
    # # manager.drop_all_tables()
    #
    # manager.calculate_distance_matrix(process_count=10)
    #
    # # manager.delete_all_data_in_table(Distance)
    p_time_start = time.time()
    # result = manager.get_photo_group_list_of_similar(0.3, 650)
    result = manager.get_group_photo_with_face(3738, 0.3, 650)
    print('Process time: ', time.time() - p_time_start)
    print('ALL-COUNT: ', len(result))
    i = 0
    for r in result:
        i += 1
    print('I: ', i)
    # manager.get_group_photo_with_face(14, 0.3)

    # # img_path = '/photo/3/3746c174a92b48ed4929dfe4ffaafda7d931629e.jpeg'
    # # img_path = '/photo/6/6ac86c8c1fae77b67adada7372464be4e941ada3.jpeg'
    # # img_path = '/photo/0/09e16386146c68fdba6b4abcb317e0c9ae5819e1.jpeg'
    # # img_path = '/photo/2/22ac1b452181f3274f3005ac49f27cb3b8100609.jpeg'
    # img_path = '/photo/2/2f74a31ac0f7d4c27e14673c7910b0c6c3078fdf.jpeg'
    # # img_path = '/photo/9/99405a718e5e9be1c3903be0180a1a4cc0c34c80.jpeg'
    # # show_recognized_faces(directory+img_path, DETECTOR_BACKEND[0], enforce_detection=True)
    #
    # img_objs = functions.extract_faces(img=directory + img_path,
    #                                    target_size=functions.find_target_size(model_name='Facenet512'),
    #                                    detector_backend='retinaface',
    #                                    grayscale=False,
    #                                    enforce_detection=True,
    #                                    align=True)
    # for img_content, img_region, img_confidence in img_objs:
    #     # # discard expanded dimension
    #     # if len(img_content.shape) == 4:
    #     #     img_content = img_content[0]
    #     # img_content = img_content[:, :, ::-1]
    #     # img = Image.fromarray((img_content * 255).astype(np.uint8))
    #     # crop_img = img.crop((img_region['x']-img_region['w']//1,
    #     #                      img_region['y']-img_region['h']//1,
    #     #                      img_region['x'] + img_region['w'] + img_region['w']//1,
    #     #                      img_region['y'] + img_region['h'] + img_region['h']//1))
    #     # crop_img = img
    #     img = Image.open(directory + img_path)
    #     # создаем новое изображение черного цвета размером в 3 раза больше чем область лица
    #     crop_img = Image.new(img.mode, (3 * img_region['w'], 3 * img_region['h']), (0, 0, 0))
    #     # в центр полученного черного квадрата вставляем область лица+1/3(области лица) с каждой стороны
    #     # иначе нераспознается лицо
    #     crop_img.paste(img.crop((img_region['x'] - img_region['w'] // 3,
    #                              img_region['y'] - img_region['h'] // 3,
    #                              img_region['x'] + img_region['w'] + img_region['w'] // 3,
    #                              img_region['y'] + img_region['h'] + img_region['h'] // 3)),
    #                    (img_region['w'] - img_region['w'] // 3, img_region['h'] - img_region['h'] // 3))
    #     crop_img.save('tmp.jpeg')
    #     resp = RetinaFace.detect_faces('tmp.jpeg')
    #     print('resp', resp)
    #
    #     draw = ImageDraw.Draw(crop_img)
    #     for i in range(1, len(resp) + 1):
    #         face = resp[f'face_{i}']
    #         draw.rectangle(face['facial_area'], outline=(255, 0, 0), width=5)
    #         draw.point(face['landmarks']['right_eye'], fill=(0, 255, 0))
    #         draw.point(face['landmarks']['left_eye'], fill=(0, 255, 0))
    #         draw.point(face['landmarks']['nose'], fill=(0, 255, 0))
    #         draw.point(face['landmarks']['mouth_left'], fill=(0, 0, 255))
    #         draw.point(face['landmarks']['mouth_right'], fill=(0, 0, 255))
    #         print(f'face_{i}', _is_point_in_tetragon(face['landmarks']['nose'],
    #                                                  face['landmarks']['mouth_right'], face['landmarks']['mouth_left'],
    #                                                  face['landmarks']['left_eye'], face['landmarks']['right_eye']))
    #     crop_img.show()
    del manager
    end_time = time.time()
    print("time in second: ", end_time - start_time, "\ntime in min: ", (end_time - start_time) / 60)

    # old time 27 min

    # start 1700 MB
    # max 9100 MB

    # t=0.2
    # 7,8,38,57,104, 217 - small face result
    # 67, 208, 235 - wrong face recognition
