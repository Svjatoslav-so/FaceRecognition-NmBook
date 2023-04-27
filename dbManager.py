import json
import multiprocessing
import time
from pathlib import Path

from deepface.commons import distance as dst
from sqlalchemy import create_engine, Column, Integer, Text, Float, String, ForeignKey, Table, Engine, event, Index, \
    UniqueConstraint, select
from sqlalchemy.orm import relationship, declarative_base, sessionmaker, Session
from tqdm import tqdm

import tool_module


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
    photo_id = Column(String(200), ForeignKey('photos.id', ondelete='CASCADE'))
    photo = relationship("Photo", back_populates='faces')

    def __repr__(self):
        return f"<Face(id:{self.id}, embedding:{self.embedding}," \
               f"x1: {self.x1}, y1: {self.y1}, x2: {self.x2}, y2: {self.y2}, " \
               f"photo_id: {self.photo_id})>"

    def to_dict(self):
        return {'id': self.id, 'embedding': self.embedding, 'x1': self.x1, 'y1': self.y1, 'x2': self.x2, 'y2': self.y2,
                'photo_id': self.photo_id}

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
    idx_face1_face2 = Index('idx_face1_face2', face1_id, face2_id)

    __table_args__ = (
        UniqueConstraint("face1_id", "face2_id"),
    )

    def __repr__(self):
        return f"<Distance(id:{self.id}, face_1:{self.face1_id}, face_2: {self.face2_id}, distance: {self.distance})>"


class DBManager:
    def __init__(self, db_name: str, echo=False):
        self.db_name = db_name
        self.engine = create_engine(f"sqlite:///{db_name}", echo=echo)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def fill_doc_photos_faces_from_file(self, embedding_file, metadata_file):
        """
            Заполняет базу данных данными из json файлов:
            embedding_file - файл с кодировками (который генерит deep_face_module_v2.all_df_encodings_to_file)
            metadata_file - файл с метаданными фото
        """
        with self.Session() as session:
            with Path(embedding_file).open(encoding="utf8") as ef, Path(metadata_file).open(encoding="utf8") as mf:
                data = json.load(ef)
                metadata = json.load(mf)['by_photo']

                for photo in data:
                    photo_obj = Photo(id=self._get_filename_from_path(photo['path']).split('.')[0],
                                      title=metadata[self._get_filename_from_path(photo['path'])]['title'])
                    for face in photo['faces']:
                        face_obj = Face(embedding=json.dumps(face['encoding']),
                                        x1=face['face_area'][0],
                                        y1=face['face_area'][1],
                                        x2=face['face_area'][2],
                                        y2=face['face_area'][3])
                        session.add(face_obj, )
                        session.flush()
                        photo_obj.faces.append(face_obj)
                    session.add(photo_obj)

                    for doc_id in metadata[self._get_filename_from_path(photo['path'])]['docs']:
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

    def get_photo_group_list_of_similar(self, threshold, min_face_area):
        """ Получаем список групп
                threshold - пороговое значение для расстояния между векторами лиц
                min_face_area - пороговое значение для площади лица
            Возвращает список словарей вида:
            {
             'photo': {'id': str, 'title': str, 'docs': [int]},
             'face':
                {'id': int, 'embedding': [float], 'x1': float, 'y1': float, 'x2': float, 'y2': float, 'photo_id': str}
            }

        """
        with self.Session() as session:
            photos_faces = session.query(Photo, Face).join(Face).filter(
                (Face.x2 - Face.x1) * (Face.y2 - Face.y1) > min_face_area
            ).join(photo_document, Face.photo_id == photo_document.c.photo_id). \
                filter(Face.id == Distance.face1_id,
                       Distance.distance < threshold,
                       Distance.face2_id < Distance.face1_id,
                       photo_document.c.doc_id.notin_(
                           select(photo_document.c.doc_id).join(Face,
                                                                Face.photo_id == photo_document.c.photo_id).filter(
                               Face.id == Distance.face2_id))
                       ).distinct().all()
            return [{'photo': p.to_dict_with_relationship(face=False), 'face': f.to_dict()} for p, f in photos_faces]

    def get_group_photo_with_face(self, origin_face, threshold, min_face_area):
        """
            Получаем фото группы
                origin_face - id лица, которое образовало группу
                threshold - пороговое значение для расстояния между векторами лиц
                min_face_area - пороговое значение для площади лица
            Возвращает список словарей вида:
            {
             'photo': {'id': str, 'title': str, 'docs': [int]},
             'face':
                {'id': int, 'embedding': [float], 'x1': float, 'y1': float, 'x2': float, 'y2': float, 'photo_id': str}
            }
         """
        with self.Session() as session:
            # origin_photo = session.query(Photo).get(origin_photo)
            origin_face = session.query(Face).get(origin_face)
            photos = session.query(Photo, Face).join(Face). \
                filter((Face.x2 - Face.x1) * (Face.y2 - Face.y1) > min_face_area). \
                join(photo_document).filter(Face.id == Distance.face2_id,
                                            # Distance.face1_id.in_([f.id for f in origin_photo.faces]),
                                            Distance.face1_id == origin_face.id,
                                            Distance.distance < threshold,
                                            Distance.face2_id < Distance.face1_id,  # ?
                                            Distance.face1_id != Distance.face2_id,
                                            photo_document.c.doc_id.notin_([d.id for d in origin_face.photo.docs])). \
                distinct().group_by(Photo.id).all()  # group_by(Photo.id)
            print(photos)
            return [{'photo': p.to_dict_with_relationship(face=False), 'face': f.to_dict()} for p, f in photos]


if __name__ == "__main__":
    print("START")
    start_time = time.time()

    directory = 'D:/Hobby/NmProject/nmbook_photo/web/static/out'
    manager = DBManager('web/db/NmBookPhoto(facenet512-retinaface).db', echo=True)
    # manager.fill_doc_photos_faces_from_file(directory + "/photo/dfv2_facenet512_encodings.json",
    #                                         directory + "/metadata.json")
    # manager.drop_all_tables()

    # manager.calculate_distance_matrix(process_count=10)

    # manager.delete_all_data_in_table(Distance)

    # manager.get_all_unique_face_group(threshold=0.08)
    # manager.test()
    manager.get_photo_group_list_of_similar(0.1)
    # manager.get_group_photo_with_face(14, 0.3)

    end_time = time.time()
    print("time in second: ", end_time - start_time, "\ntime in min: ", (end_time - start_time) / 60)

    # old time 27 min

    # start 1700 MB
    # max 9100 MB
