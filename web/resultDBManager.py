import time

from sqlalchemy import Column, Integer, Text, Float, String, ForeignKey, Engine, event, Boolean, Index, \
    UniqueConstraint, create_engine, select
from sqlalchemy.orm import relationship, declarative_base, sessionmaker


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
    photo_id = Column(String(200), nullable=False)
    x1 = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    x2 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)
    photo_title = Column(Text)
    docs = Column(Text, nullable=False)

    __table_args__ = (
        UniqueConstraint("photo_id", "x1", "y1", "x2", "y2"),
    )

    def __repr__(self):
        return f"<Face(id:{self.id}, photo_id:{self.photo_id}, " \
               f"x1: {self.x1}, y1: {self.y1}, x2: {self.x2}, y2: {self.y2}, " \
               f"photo_title: {self.photo_title}, " \
               f"docs: {self.docs})>"

    def to_dict(self):
        return {'id': self.id, 'photo_id': self.photo_id, 'x1': self.x1, 'y1': self.y1, 'x2': self.x2, 'y2': self.y2,
                'photo_title': self.photo_title, 'docs': self.docs}

    def to_dict_with_relationship(self):
        dic = self.to_dict()
        return dic


class Bookmark(Base):
    __tablename__ = 'bookmarks'
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False, unique=True)
    origin_id = Column(Integer, ForeignKey('faces.id'), nullable=False)
    similar_id = Column(Integer, ForeignKey('faces.id'), nullable=False)
    db = Column(Text)  # комментарий к закладке
    origin = relationship("Face", primaryjoin="Bookmark.origin_id == Face.id")
    similar = relationship("Face", primaryjoin="Bookmark.similar_id == Face.id")
    idx_face1_face2_distance = Index('idx_origin_similar_db', origin_id, similar_id, db)

    __table_args__ = (
        UniqueConstraint("origin_id", "similar_id"),
    )

    def __repr__(self):
        return f"<Bookmark(id:{self.id}, origin_id:{self.origin_id}, similar_id: {self.similar_id}, db: {self.db})>"


class ResultDBManager:
    def __init__(self, db_name: str, echo=False):
        self.db_name = db_name
        self.engine = create_engine(f"sqlite:///{db_name}", echo=echo)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def get_or_create_face(self, photo_id: str, x1: float, y1: float, x2: float, y2: float, photo_title: str,
                           docs: str):
        """
            Возвращает из базы-данных объект класса Face с такими же photo_id, x1, y1, x2 и y2.
            Если такой записи нет, то она предварительно будет добавлена.
        """
        with self.Session() as session:
            session.expire_on_commit = False
            face = session.query(Face).filter(Face.photo_id == photo_id, Face.x1 == x1, Face.y1 == y1,
                                              Face.x2 == x2, Face.y2 == y2).first()
            if not face:
                face = Face(photo_id=photo_id, x1=x1, y1=y1, x2=x2,
                            y2=y2, photo_title=photo_title, docs=docs)
                session.add(face)
                session.commit()
            return face

    def add_bookmarks(self,
                      origin: dict['photo_id': str, 'x1': float, 'y1': float, 'x2': float, 'y2': float,
                              'photo_title': str, 'docs': str],
                      similar: list[dict['photo_id': str, 'x1': float, 'y1': float, 'x2': float, 'y2': float,
                                    'photo_title': str, 'docs': str]],
                      comment: str = ''):
        """
            Добавляет закладки.
            Для origin-лица ставятся в соответствие лица из similar. comment - комментарий к закладкам
        """
        with self.Session() as session:
            origin_face = self.get_or_create_face(photo_id=origin['photo_id'],
                                                  x1=origin['x1'], y1=origin['y1'], x2=origin['x2'], y2=origin['y2'],
                                                  photo_title=origin['photo_title'], docs=origin['docs'])
            new_bookmark = []
            for s in similar:
                print('S befor:', s)
                s = self.get_or_create_face(photo_id=s['photo_id'],
                                            x1=s['x1'], y1=s['y1'], x2=s['x2'], y2=s['y2'],
                                            photo_title=s['photo_title'], docs=s['docs'])
                print('Origin Face: ', origin_face)
                print('Similar Face: ', s)

                bookmark = session.query(Bookmark).filter(Bookmark.origin_id == origin_face.id,
                                                          Bookmark.similar_id == s.id).first()
                if not bookmark:
                    bookmark = Bookmark(origin_id=origin_face.id, similar_id=s.id, db=comment)
                    new_bookmark.append(bookmark)
            session.add_all(new_bookmark)
            session.commit()

    def get_all_similar_of(self, origin_photo_id: str, origin_face_x1: float, origin_face_y1: float,
                           origin_face_x2: float, origin_face_y2: float):
        """
            Возвращает список лиц находящихся в закладках с лицом у которого
            photo_id == origin_photo_id,
            x1 == origin_face_x1,
            y1 == origin_face_y1,
            x2 == origin_face_x2,
            y2 == origin_face_y2,
        """
        with self.Session() as session:
            origin_face = session.query(Face).filter(Face.photo_id == origin_photo_id,
                                                     Face.x1 == origin_face_x1, Face.y1 == origin_face_y1,
                                                     Face.x2 == origin_face_x2, Face.y2 == origin_face_y2).first()
            if not origin_face:
                return []

            similar_faces = session.query(Face).join(Bookmark, Face.id == Bookmark.similar_id).filter(
                Bookmark.origin_id == origin_face.id).all()
            return [s.to_dict() for s in similar_faces]

    def delete_bookmarks(self,
                         origin: dict['photo_id': str, 'x1': float, 'y1': float, 'x2': float, 'y2': float,
                                 'photo_title': str, 'docs': str],
                         similar: list[dict['photo_id': str, 'x1': float, 'y1': float, 'x2': float, 'y2': float,
                                       'photo_title': str, 'docs': str]]):
        """
            Если существует origin, то из закладок связанных с ним удаляем связи с лицами из similar (если они
            существуют в бд).
            После этого удаляются все лица которые не имеют связей в bookmarks (ни в качестве origin, ни similar).
        """
        with self.Session() as session:
            origin_face = session.query(Face).filter(Face.photo_id == origin['photo_id'], Face.x1 == origin['x1'],
                                                     Face.y1 == origin['y1'], Face.x2 == origin['x2'],
                                                     Face.y2 == origin['y2']).first()
            if not origin_face:
                print('ORIGIN EXIST')
                return

            for s in similar:
                print('S befor:', s)
                s = session.query(Face).filter(Face.photo_id == s['photo_id'],
                                               Face.x1 == s['x1'], Face.y1 == s['y1'], Face.x2 == s['x2'],
                                               Face.y2 == s['y2']).first()
                if not s:
                    print('SIMILAR EXIST')
                    continue
                else:
                    bookmark = session.query(Bookmark).filter(Bookmark.origin_id == origin_face.id,
                                                              Bookmark.similar_id == s.id).first()
                    if bookmark:
                        print('BOOKMARK EXIST')
                        session.delete(bookmark)
                        session.commit()
            session.query(Face).filter(Face.id.notin_(select(Bookmark.origin_id)),
                                       Face.id.notin_(select(Bookmark.similar_id))).delete()
            session.commit()

    def get_all_bookmark_groups(self):
        """ Возвращает список лиц, id которых в таблице bookmarks стоит в качестве origin_id """
        with self.Session() as session:
            groupmaker_faces = session.query(Face).filter(Face.id.in_(select(Bookmark.origin_id))).all()
            return [gf.to_dict() for gf in groupmaker_faces]

    def get_comment_for_bookmark(self, origin_photo_id: str, origin_face_x1: float, origin_face_y1: float,
                                 origin_face_x2: float, origin_face_y2: float, similar_photo_id: str,
                                 similar_face_x1: float, similar_face_y1: float,
                                 similar_face_x2: float, similar_face_y2: float):
        with self.Session() as session:
            origin_id = session.query(Face.id).filter(
                                                      Face.photo_id == origin_photo_id,
                                                      Face.x1 == origin_face_x1,
                                                      Face.y1 == origin_face_y1,
                                                      Face.x2 == origin_face_x2,
                                                      Face.y2 == origin_face_y2)
                                                      # Face.y2 == origin_face_y2).first()
            # print('origin_id: ', origin_id)
            similar_id = session.query(Face.id).filter(
                                                       Face.photo_id == similar_photo_id,
                                                       Face.x1 == similar_face_x1,
                                                       Face.y1 == similar_face_y1,
                                                       Face.x2 == similar_face_x2,
                                                       Face.y2 == similar_face_y2)
                                                       # Face.y2 == similar_face_y2).first()
            # print('similar_id: ', similar_id)
            comment = session.query(Bookmark.db).filter(
                                                    Bookmark.origin_id.in_(origin_id),
                                                    Bookmark.similar_id.in_(similar_id)
                                                    ).first()
            return comment[0] if comment and len(comment) else comment


if __name__ == "__main__":
    print("START")
    start_time = time.time()
    manager = ResultDBManager('NmBookPhoto-result.db')

    # manager.add_bookmarks(
    #     origin={'photo_id': '001385c7cfe3784ab9073d47a8088df015121838', 'x1': 489, 'y1': 844, 'x2': 508, 'y2': 868,
    #             'photo_title': 'Священник Благовещенский Алексей Петрович', 'docs': '29'},
    #     similar=[
    #         {'photo_id': '0082eb3bcfdfa0488dadc7fe1b33625d8bbd0c7e', 'x1': 234, 'y1': 112, 'x2': 384,
    #          'y2': 363,
    #          'photo_title': 'Протоиерей Василий (Лихарев Василий Алексеевич), тюремная фотография.',
    #          'docs': '69'},
    #         {'photo_id': '00ed027911b0c2a9e199eecdbac3482b585da145', 'x1': 122, 'y1': 145, 'x2': 239,
    #          'y2': 306,
    #          'photo_title': 'Групповая фотография, 25 октября 1927 г.: Василий Павлович Коклин во втором ряду третий слева',
    #          'docs': '29'},
    #         {'photo_id': '00c192ad83bcb0025e7146d9fec5f442bdc45b71', 'x1': 1030, 'y1': 176, 'x2': 1148,
    #          'y2': 336,
    #          'photo_title': 'Игумен Дамаскин (Жабинский) и диакон Иосиф Потапов у гроба Матроны Андреевны Сахаровой. 31 ноября 1930 г.',
    #          'docs': '29'}
    #     ],
    #     comment='')

    #print(manager.get_comment_for_bookmark('8d3a62e4f6ec49b86784c0030344305a69e8ff5e',129,124,253,274,
    #                                 '3cbbd20f17e6b3c9185ca8d5007b8ba642143f3c', 235, 51, 285, 114))

    end_time = time.time()
    print("time in second: ", end_time - start_time, "\ntime in min: ", (end_time - start_time) / 60)
