import torch
import redis
import io  # Used for binary serialization
import hashlib
import numpy as np
import chromadb


def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


class StartRecentKVCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        redis_host='localhost',
        redis_port=6379,
        redis_db=0
    ):
        print(f"StartRecentKVCache: {start_size}, {recent_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        
        # Connect to Redis
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        self.chroma = chromadb.Client()
        self.collection = self.chroma.create_collection(name="embeddings")


    # def __call__(self, past_key_values):
    #     if past_key_values is None:
    #         return None
    #     seq_len = past_key_values[0][0].size(self.k_seq_dim)
    #     if seq_len <= self.cache_size:
    #         return past_key_values
    #     return [
    #         [
    #             torch.cat(
    #                 [
    #                     self.k_slice(k, 0, self.start_size),
    #                     self.k_slice(k, seq_len - self.recent_size, seq_len),
    #                 ],
    #                 dim=self.k_seq_dim,
    #             ),
    #             torch.cat(
    #                 [
    #                     self.v_slice(v, 0, self.start_size),
    #                     self.v_slice(v, seq_len - self.recent_size, seq_len),
    #                 ],
    #                 dim=self.v_seq_dim,
    #             ),
    #         ]
    #         for k, v in past_key_values
    #     ]

    def clear_db(self):
        self.redis_client.flush()

    def concat(self, kv1, kv2):
        if not kv2:
            return kv1
        if not kv1:
            return kv2
        # print(len(kv1), len(kv2))
        
        return [
                [
                    torch.cat(
                        [
                            k1, k2
                        ],
                        dim=self.k_seq_dim,
                    ),
                    torch.cat(
                        [
                            v1, v2
                        ],
                        dim=self.v_seq_dim,
                    ),
                ]
                for (k1, v1), (k2, v2) in zip(kv1, kv2)
            ]

    def evict_for_space(self, past_key_values, num_coming, generated_text):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        # print("cur cache_size, ", self.cache_size)
        # print(f'seq{seq_len}, coming{num_coming}')

        stored_kvs = [
            [
                self.k_slice(k, seq_len -  num_coming, seq_len),
                self.v_slice(v, seq_len -  num_coming, seq_len),
            ]
            for k, v in past_key_values
        ]
        # Store evicted KVs in Redis


        text = " ".join(generated_text)
        redis_key = hashlib.sha256(text.encode()).hexdigest()
        # print("added text", text)
        self.collection.add(documents=[
            text
        ], ids=[redis_key]
        )

        self.store_in_redis(redis_key, stored_kvs, 'evict_for_space')

        if seq_len + num_coming <= self.cache_size:
            return past_key_values

        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(
                            k, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(
                            v, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def get_doc(self, text):
        res = self.collection.query(
            query_texts = [text],
            n_results = 1
        )
        return res["documents"][0]

    def get_rag(self, text):
        res = self.collection.query(
            query_texts = [text],
            n_results = 1
        )
        kvs = []
        for ids in res["ids"][0]:
            kvs.extend(self.retrieve_in_redis(ids))
        return kvs
        # return []
    
    def retrieve_in_redis(self, redis_key):
        batch_data = self.redis_client.hgetall(redis_key)
        evicted_kvs = []
        idx = 0
        while f'k_{idx}'.encode() in batch_data:
            k = self.deserialize_tensor(batch_data[f'k_{idx}'.encode()])
            v = self.deserialize_tensor(batch_data[f'v_{idx}'.encode()])
            evicted_kvs.append([k, v])
            idx += 1
        return evicted_kvs
        
    # def evict_range(self, past_key_values, start, end):
    #     if past_key_values is None:
    #         return None
    #     seq_len = past_key_values[0][0].size(self.k_seq_dim)
    #     assert start <= end and end <= seq_len
        
    #     evicted_kvs = [
    #         [
    #             self.k_slice(k, start, end),
    #             self.v_slice(v, start, end),
    #         ]
    #         for k, v in past_key_values
    #     ]
        
    #     # Store evicted KVs in Redis
    #     self.store_in_redis(evicted_kvs, f'evict_range_{start}_{end}')
        
    #     return [
    #         [
    #             torch.cat(
    #                 [
    #                     self.k_slice(k, 0, start),
    #                     self.k_slice(k, end, seq_len),
    #                 ],
    #                 dim=self.k_seq_dim,
    #             ),
    #             torch.cat(
    #                 [
    #                     self.v_slice(v, 0, start),
    #                     self.v_slice(v, end, seq_len),
    #                 ],
    #                 dim=self.v_seq_dim,
    #             ),
    #         ]
    #         for k, v in past_key_values
    #     ]

    def store_in_redis(self, redis_key, evicted_kvs, eviction_type):
        """
        Store evicted key-value pairs in Redis. Each key-value pair is serialized 
        and stored with a unique key in the Redis database.
        
        Args:
            evicted_kvs: The list of evicted key-value tensors.
            eviction_type: A string identifying the type of eviction ('evict_for_space' or 'evict_range').
        """
        try:
            batch_serialized = {}
            for idx, (k, v) in enumerate(evicted_kvs):
                k_serialized = self.serialize_tensor(k)
                v_serialized = self.serialize_tensor(v)

                batch_serialized[f'k_{idx}'] = k_serialized
                batch_serialized[f'v_{idx}'] = v_serialized
            
            self.redis_client.hset(redis_key, mapping=batch_serialized)
            # print("Stored")
        except Exception as e:
            print(f"Failed to store batch of KV pairs in Redis: {e}")
            return None


    def tensor_to_redis_key(self, tensor):
        """
        Generate a unique Redis key by hashing the tensor's contents.
        """
        tensor_bytes = self.serialize_tensor(tensor)  # Serialize the tensor to binary
        # Hash the binary data to get a unique key
        return hashlib.sha256(tensor_bytes).hexdigest()

    @staticmethod
    def serialize_tensor(tensor):
        """
        Serialize a PyTorch tensor to a binary format using torch.save.
        
        Args:
            tensor: The PyTorch tensor to serialize.
        
        Returns:
            The binary-encoded tensor.
        """
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        return buffer.getvalue()

    @staticmethod
    def deserialize_tensor(binary_data):
        """
        Deserialize a binary-encoded PyTorch tensor.
        
        Args:
            binary_data: The binary-encoded tensor.
        
        Returns:
            The deserialized PyTorch tensor.
        """
        buffer = io.BytesIO(binary_data)
        return torch.load(buffer)
