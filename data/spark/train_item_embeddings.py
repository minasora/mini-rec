import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import Normalizer
from pyspark.ml.functions import array_to_vector
from pyspark.sql.types import FloatType
from pyspark.ml.linalg import DenseVector

# 1. 初始化 SparkSession
spark = SparkSession.builder \
    .appName("MiniItemEmb_with_array_to_vector") \
    .getOrCreate()

# 2. 加载并过滤评分（仅保留 rating >= 3.5）
ratings = (
    spark.read
         .option("header", True)
         .csv("data/ratings.csv")
         .select(
             F.col("userId").cast("int"),
             F.col("movieId").cast("int"),
             F.col("rating").cast("float")
         )
         .where("rating >= 3.5")
)

# 可选：将所有正反馈统一为 1.0（隐式反馈场景）
positive = ratings.withColumn("rating", F.lit(1.0))

# 3. 训练 ALS 模型
als = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    rank=64,
    maxIter=10,
    regParam=0.1,
    implicitPrefs=True,
    coldStartStrategy="drop"
)
model = als.fit(positive)

# 4. 提取 raw item factors (movieId, features:Array<Float>)
item_factors = model.itemFactors

# 5. 把 Array<Float> → ML VectorUDT
item_vecs = item_factors.withColumn(
    "features",
    array_to_vector("features")
)

# 6. L2 归一化得到单位向量，用于余弦相似度
normaliser = Normalizer(inputCol="features", outputCol="norm")
# —— 注意：这里 transform 要传 item_vecs 而不是 item_factors ——
item_norm = normaliser.transform(item_vecs)

# 7. 将结果写出到 Parquet
item_norm.write.mode("overwrite").parquet("data/item_embeddings.parquet")
print("✅ Saved normalized item embeddings.")

# ———— 简单的相似度查询示例 ————

# 随便挑一个 movieId
TARGET_ID = 80

# 取出目标向量
target_vec = (
    item_norm
    .where(F.col("id") == TARGET_ID)
    .select("norm")
    .first()["norm"]  # DenseVector
)

# 用 UDF 计算点积（单位向量的点积即余弦相似度）
dot_udf = F.udf(lambda v: float(v.dot(target_vec)), FloatType())

top_similar = (
    item_norm
    .withColumn("score", dot_udf("norm"))
    .where(F.col("id") != TARGET_ID)
    .orderBy(F.desc("score"))
    .limit(10)
)

print("Top similar items:")
top_similar.show(truncate=False)

# 8. 结束 Spark
spark.stop()
