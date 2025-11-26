"""
MongoDB 클라이언트 (메타데이터 저장)
"""
import logging
from typing import Any
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from app.vecdb.interfaces import MetadataStore


logger = logging.getLogger(__name__)


class MongoDBClient(MetadataStore):
    """
    MongoDB 클라이언트
    - 문서 메타데이터 저장
    - 청크 정보 관리
    """
    
    def __init__(
        self,
        connection_string: str = "mongodb://localhost:27017/",
        database_name: str = "semiconductor_rag",
        collection_name: str = "documents"
    ):
        """
        Args:
            connection_string: MongoDB 연결 문자열
            database_name: 데이터베이스 이름
            collection_name: 컬렉션 이름
        """
        self.connection_string = connection_string
        self.database_name = database_name
        self.collection_name = collection_name
        
        try:
            self.client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            self.client.server_info()
            
            self.db = self.client[database_name]
            self.collection = self.db[collection_name]
            
            self._create_indexes()
            
            logger.info(f"✓ MongoDB 연결 성공")
            logger.info(f"  - Database: {database_name}")
            logger.info(f"  - Collection: {collection_name}")
            logger.info(f"  - 문서 수: {self.count()}")
            
        except PyMongoError as e:
            logger.error(f"MongoDB 연결 실패: {e}")
            raise
    
    def _create_indexes(self) -> None:
        """인덱스 생성"""
        try:
            self.collection.create_index("chunk_id", unique=True)
            self.collection.create_index("paper_filename")
            self.collection.create_index("domain")
            self.collection.create_index([("created_at", -1)])
            
            logger.debug("MongoDB 인덱스 생성 완료")
        except PyMongoError as e:
            logger.warning(f"인덱스 생성 실패: {e}")
    
    def insert_many(self, documents: list[dict[str, Any]]) -> list[str]:
        """
        다중 문서 삽입
        
        Returns:
            삽입된 문서 ID 리스트
        """
        if not documents:
            return []
        
        for doc in documents:
            doc['created_at'] = doc.get('created_at', datetime.now())
            doc['updated_at'] = datetime.now()
        
        try:
            result = self.collection.insert_many(documents, ordered=False)
            logger.info(f"MongoDB 문서 삽입: {len(result.inserted_ids)}개")
            return [str(doc_id) for doc_id in result.inserted_ids]
        except PyMongoError as e:
            logger.error(f"문서 삽입 실패: {e}")
            return []
    
    def find_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """ID로 검색"""
        try:
            from bson.objectid import ObjectId
            object_ids = [ObjectId(doc_id) for doc_id in ids if ObjectId.is_valid(doc_id)]
            
            cursor = self.collection.find({'_id': {'$in': object_ids}})
            documents = list(cursor)
            
            for doc in documents:
                doc['_id'] = str(doc['_id'])
            
            return documents
        except Exception as e:
            logger.error(f"문서 검색 실패: {e}")
            return []
    
    def find_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        """청크 ID로 검색"""
        try:
            cursor = self.collection.find({'chunk_id': {'$in': chunk_ids}})
            documents = list(cursor)
            
            for doc in documents:
                doc['_id'] = str(doc['_id'])
            
            return documents
        except Exception as e:
            logger.error(f"청크 검색 실패: {e}")
            return []
    
    def update_one(self, doc_id: str, update_data: dict[str, Any]) -> bool:
        """문서 업데이트"""
        try:
            from bson.objectid import ObjectId
            
            if not ObjectId.is_valid(doc_id):
                return False
            
            update_data['updated_at'] = datetime.now()
            
            result = self.collection.update_one(
                {'_id': ObjectId(doc_id)},
                {'$set': update_data}
            )
            
            return result.modified_count > 0
        except PyMongoError as e:
            logger.error(f"문서 업데이트 실패: {e}")
            return False
    
    def count(self) -> int:
        """문서 수 반환"""
        try:
            return self.collection.count_documents({})
        except PyMongoError as e:
            logger.error(f"문서 카운트 실패: {e}")
            return 0
    
    def get_stats(self) -> dict[str, Any]:
        """통계 반환"""
        try:
            pipeline = [
                {
                    '$group': {
                        '_id': '$domain',
                        'count': {'$sum': 1}
                    }
                }
            ]
            
            domain_stats = list(self.collection.aggregate(pipeline))
            
            return {
                'total_documents': self.count(),
                'domains': {stat['_id']: stat['count'] for stat in domain_stats},
                'database': self.database_name,
                'collection': self.collection_name,
            }
        except Exception as e:
            logger.error(f"통계 조회 실패: {e}")
            return {}
    
    def close(self) -> None:
        """연결 종료"""
        if self.client:
            self.client.close()
            logger.info("MongoDB 연결 종료")




