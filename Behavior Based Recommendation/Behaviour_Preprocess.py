import numpy as np
import pandas as pd
import sys
import math
import glob as gb
import pickle
import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn import preprocessing
from scipy import sparse
from pyspark.sql import Row
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StringIndexer
import sys


#Spart set up
conf = SparkConf() \
    .setAppName("DeepLearningRecommendation") \
    .set("spark.executor.memory", "12g")\
    .set("spark.driver.memory", "10g")\
    .set("spark.sql.pivotMaxValues","100000")\
    .set("spark.debug.maxToStringFields","30000")\
    .set("spark.sql.shuffle.partitions","400")\
    .set("spark.sql.files.maxPartitionBytes", 32000000)
sc = SparkContext(conf=conf)
spark = SQLContext(sc)

def dataNormalization(Order_History):
    Order_History = Order_History.reset_index()
    Order_History_Normal = Order_History[["Quantity"]]
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    scaled = min_max_scaler.fit_transform(Order_History_Normal)
    Order_History_Normalized = pd.DataFrame(scaled)
    Order_History["Quantity"] = Order_History_Normalized[[0]]
    return Order_History
def split(arr, size):
    arrs = []
    while len(arr) > size:
        pice = arr[:size]
        arrs.append(pice)
        arr = arr[size:]
    arrs.append(arr)
    return arrs

def returnNotMatches(a, b):
    a = set(a)
    b = set(b)
    return list(b-a),list(a-b)


def missingProducts(product_Feature,Order_History):
    product_from_order = list(set(Order_History['Product_Id'].tolist()))
    product_List = list(set(product_Feature['Unnamed: 0'].tolist()))
    product_not_in_feature, product_not_in_order = returnNotMatches(product_List, product_from_order)
    Order_History = Order_History.drop(
        Order_History[Order_History.Product_Id.isin(product_not_in_feature)].index.tolist())
    Order_History =Order_History.reset_index(drop=True)
    return Order_History

def behaviourDataProcess(file_path,product_feature):
    #file_path = "D:\\RecommendationEngine\\Orders\\Orders\\*.csv"
    Order_History_List = []
    for DataFrame in gb.glob(file_path):
        print(DataFrame)
        Order_History = pd.read_csv(DataFrame, encoding="ISO-8859-1", header=0)
        Order_History_temp = Order_History.loc[:,['MEMBER_ID', 'PRODUCTPARTNUMBER', 'TOTAL']]
        Order_History_List.append(Order_History_temp)
    Order_History_temp = pd.concat(Order_History_List, axis=0, ignore_index=True)
    del Order_History_List
    Order_History_temp.rename(columns={Order_History_temp.columns[0]: 'Customer_Id'}, inplace=True)
    Order_History_temp.rename(columns={Order_History_temp.columns[1]: 'Product_Id'}, inplace=True)
    Order_History_temp.rename(columns={Order_History_temp.columns[2]: 'Quantity'}, inplace=True)
    Order_History_temp = missingProducts(product_feature,Order_History_temp)
    Order_History = Order_History_temp.groupby(["Customer_Id", "Product_Id"])["Quantity"].sum()
    del Order_History_temp
    Order_History = Order_History.to_frame().reset_index()
    Order_History.index.name = 'index'
    Order_History = Order_History.dropna()
    #del Order_History_temp

    #Spark Process
    Order_History=dataNormalization(Order_History)
    Order_History=Order_History.dropna()
    print(len(Order_History[['Customer_Id']]))
    Order_History = spark.createDataFrame(Order_History)
    Order_History = Order_History.drop('index')
    #print(Order_History.show(10))
    Order_History =Order_History.withColumn("Quantity",Order_History["Quantity"].cast(DoubleType()))
    Order_History = Order_History.groupBy('Customer_Id').pivot("Product_Id").sum("Quantity").fillna(0)
    # indexer = StringIndexer(inputCol="Customer_Id", outputCol="id")
    # Order_History = indexer.fit(Order_History).transform(Order_History)
    # print("before indexing is")
    # Order_History=Order_History.withColumn("id", Order_History["id"].cast(DoubleType()))
    print(len(Order_History.columns))
    #print(len(Order_History[["Customer_Id"]]))
    #print(sys.getsizeof(Order_History))
    Columns = Order_History.columns
    Columns.sort()
    #Columns.remove('id')
    Columns.remove('Customer_Id')
    Columns = split(Columns,1720)
    print(len(Columns))
    count = 0
    #Order_History.persist(pyspark.StorageLevel.MEMORY_ONLY)
    for C in Columns:
        #C.append('id')
        C.append("Customer_Id")
        Order_History_temp = Order_History.select(*tuple(C))
        Order_History_temp.toPandas().to_csv("/home/sakthi10feb88/Tempyy_Order_History" + str(count) + "_0"  + ".csv")
                                      #        ,
                                      # sep=',',
                                      # header=True)
        count += 1

        # # Order_History_temp = Order_History.select(*tuple(C))
        # Order_History_temp1 = Order_History.filter(Order_History.id <= 30000)
        # Order_History_temp1.drop('id')
        # # Order_History_temp1.rdd.map(lambda x: ",".join(map(str, x))).coalesce(1).saveAsTextFile("/home/sakthi10feb88/Output/Order_History" + str(count) + "_"+str(10) + ".csv")
        # Order_History_temp1.write.csv("/home/sakthi10feb88/Order_History" + str(count) + "_" + str(10) + ".csv", sep=',',
        #                               header=True)
        # print("1st import is done")
        # del Order_History_temp1
        # Order_History_temp2 = Order_History.filter((Order_History.id > 30000)& Order_History.id <= 60000 )
        # Order_History_temp2.drop('id')
        # # Order_History_temp1.rdd.map(lambda x: ",".join(map(str, x))).coalesce(1).saveAsTextFile(
        # #     "/home/sakthi10feb88/Output/Order_History" + str(count) + "_" + str(20) + ".csv")
        # Order_History_temp2.write.csv("/home/sakthi10feb88/Order_History" + str(count) + "_"+ str(20) + ".csv", sep=',',
        #                               header=True)
        # print("2nd import is done")
        # del Order_History_temp2
        # Order_History_temp3 = Order_History.filter(Order_History.id > 60000)
        # Order_History_temp3.drop('id')
        # # Order_History_temp1.rdd.map(lambda x: ",".join(map(str, x))).coalesce(1).saveAsTextFile(
        # #     "/home/sakthi10feb88/Output/Order_History" + str(count) + "_" + str(30) + ".csv")
        # Order_History_temp3.write.csv("/home/sakthi10feb88/Order_History" + str(count) +"_"+ str(30) + ".csv", sep=',',
        #                               header=True)
        # print("3rd import is done")
        # del Order_History_temp3
        # count+=1

    # Order_History = Order_History.toPandas()
    # Order_History = Order_History.set_index("Customer_Id")
    # del Order_History.index.name
    # Order_History = sparse.coo_matrix(Order_History)
    #return Order_History

if __name__ =='__main__':
    file_path = "D:\\RecommendationEngine\\Orders\\Orders\\*.csv"
    product_feature=pd.read_csv("product_Feature.csv", encoding="ISO-8859-1",header=0)
    behaviourDataProcess(file_path,product_feature)
    # with open("Behaviour_Sparse_Matrix.pickle",'wb') as handle:
    #     pickle.dump(Order_History, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #Order_History.to_csv("/home/sakthi10feb88/Order_History.csv")
    # Order_History.to_csv("Order_History.csv")

