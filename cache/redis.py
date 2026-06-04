import hashlib
import json
import redis
from config.settings import settings


class SearchCache:
    def __init__(self):
        self.redis = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True,
        )
    
    def get(self, query, method):
        key = hashlib.md5(f"{query}:{method}".encode()).hexdigest()
        data = self.redis.get(f"search:{method}:{key}")
        
        return json.loads(data) if data else None
    
    def set(self, query, method, results, ttl = 900):
        key = hashlib.md5(f"{query}:{method}".encode()).hexdigest()
        self.redis.setex(f"search:{method}:{key}", ttl, json.dumps(results))
        