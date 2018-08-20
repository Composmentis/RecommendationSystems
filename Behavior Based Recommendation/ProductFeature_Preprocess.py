import tensorrec
import tensorflow as tf
import pandas as pd
import glob as gb
import pickle
from scipy import sparse
from sklearn import preprocessing

def pivotingData(dataFrame,indexColumn,columnsColumn,valuesColumn):
    return dataFrame.pivot_table(index=indexColumn, columns=columnsColumn, values=valuesColumn)

def splitDataFrameIntoSmaller(df, chunkSize = 5000):
    listOfDf = list()
    numberChunks = len(df) // chunkSize + 1
    for i in range(numberChunks):
        listOfDf.append(df[i*chunkSize:(i+1)*chunkSize])
    return listOfDf

def dataNormalization(Order_History):
    Order_History_Normal = Order_History[["Quantity"]]
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    scaled = min_max_scaler.fit_transform(Order_History_Normal)
    Order_History_Normalized = pd.DataFrame(scaled)
    Order_History[["Quantity"]] = Order_History_Normalized[[0]]
    return Order_History

def cleaningProductFeature(file_path,category_to_be_considered):
    # file_path = "D:\\RecommendationEngine\\CatalogData_Lates\\product_attributes.csv"
    product_Feature = pd.read_csv(file_path, encoding="ISO-8859-1", header=0)
    List_Temp = set(product_Feature['ProductType'].tolist())
    category_to_be_not_considered = list(List_Temp - set(category_to_be_considered))
    product_Feature = product_Feature.drop(
        product_Feature[product_Feature.ProductType.isin(category_to_be_not_considered)].index.tolist())
    product_Feature = product_Feature.reset_index(drop=True)
    product_Feature.loc[product_Feature['CareInstruction'].isna(), "WithOutCareInstruction"] = 1
    product_Feature['WithOutCareInstruction'] = product_Feature['WithOutCareInstruction'].fillna(0)
    product_Feature.loc[product_Feature['Gender'].isna(), "WithOutGender"] = 1
    product_Feature['WithOutGender'] = product_Feature['WithOutGender'].fillna(0)
    product_Feature['CareInstruction'] = product_Feature['CareInstruction'].fillna(
        product_Feature['CareInstruction'].value_counts().index[0])
    product_Feature['Gender'] = product_Feature['Gender'].fillna(product_Feature['Gender'].value_counts().index[0])
    product_Feature['SellingPriceStructure'] = product_Feature['SellingPriceStructure'].fillna(
        product_Feature['SellingPriceStructure'].value_counts().index[0])
    product_Feature['DisplayStyleWithIt'] = product_Feature['DisplayStyleWithIt'].fillna(
        product_Feature['DisplayStyleWithIt'].value_counts().index[0])
    product_Feature['values_for_pivoting'] = 1
    return product_Feature

def preProcessingProductFeature(file_path,file_path_orders,category_to_be_considered):
    product_Feature = cleaningProductFeature(file_path,category_to_be_considered)
    product_Feature_List = splitDataFrameIntoSmaller(product_Feature, 1000)
    product_Feature_temp_List = []
    for product in product_Feature_List:
        temp = pivotingData(product, "Product_Id", "CareInstruction", "values_for_pivoting")
        product_Feature_temp_List.append(temp.reset_index())
    product_Feature_1 = pd.concat(product_Feature_temp_List, axis=0, ignore_index=True)
    product_Feature_1 = pd.merge(product_Feature_1,product_Feature.loc[:,["Product_Id","WithOutCareInstruction","WithOutGender",
                                                   "MaximumDaysOutOfStockBeforeHi","DisplayStyleWithIt"]], on='Product_Id', how='outer')
    del product_Feature_temp_List

    # product_Feature_temp_List = []
    # for product in product_Feature_List:
    #     temp = pivotingData(product, "Product_Id", "SellingPriceStructure", "values_for_pivoting")
    #     product_Feature_temp_List.append(temp.reset_index())
    # product_Feature_2 = pd.concat(product_Feature_temp_List, axis=0, ignore_index=True)
    # product_Feature_1 = pd.merge(product_Feature_1, product_Feature_2, on='Product_Id', how='outer')
    # del product_Feature_2,product_Feature_temp_List

    product_Feature_temp_List = []
    for product in product_Feature_List:
        temp = pivotingData(product, "Product_Id", "Gender", "values_for_pivoting")
        product_Feature_temp_List.append(temp.reset_index())
    product_Feature_3 = pd.concat(product_Feature_temp_List, axis=0, ignore_index=True)
    product_Feature_1 = pd.merge(product_Feature_1, product_Feature_3, on='Product_Id', how='outer')
    del product_Feature_3,product_Feature_temp_List

    product_Feature_temp_List = []
    for product in product_Feature_List:
        temp = pivotingData(product, "Product_Id", "ProductType", "values_for_pivoting")
        product_Feature_temp_List.append(temp.reset_index())
    product_Feature_4 = pd.concat(product_Feature_temp_List, axis=0, ignore_index=True)
    product_Feature = pd.merge(product_Feature_1, product_Feature_4, on='Product_Id', how='outer')
    del product_Feature_4,product_Feature_1,product_Feature_List,product_Feature_temp_List
    product_Feature = missingProducts(product_Feature,file_path_orders)
    product_Feature = product_Feature.fillna(0)
    product_Feature = product_Feature.sort_values("Product_Id")
    product_Feature=product_Feature.set_index("Product_Id")
    del product_Feature.index.name
    product_Feature.to_csv("product_Feature.csv")
    print(product_Feature.shape)
    print(product_Feature.columns)
    product_Feature = sparse.lil_matrix(product_Feature)
    print(product_Feature.shape)
    return product_Feature

def missingProducts(product_Feature,file_path_orders):
    #file_path_orders = "D:\\RecommendationEngine\\Orders\\Orders\\*.csv"
    product_from_order = []
    for DataFrame in gb.glob(file_path_orders):
        Order_History = pd.read_csv(DataFrame, encoding="ISO-8859-1", header=0)
        Order_History = Order_History['PRODUCTPARTNUMBER'].tolist()
        product_from_order = product_from_order + Order_History
    product_from_order = set(product_from_order)
    product_from_order = list(product_from_order)
    # print(len(product_from_order))
    product_List = (list(set(product_Feature['Product_Id'].tolist())))
    product_not_in_feature, product_not_in_order = returnNotMatches(product_List, product_from_order)
    # print(product_Feature.head(10))
    product_Feature = product_Feature.drop(
    product_Feature[product_Feature.Product_Id.isin(product_not_in_order)].index.tolist())
    product_Feature = product_Feature.reset_index(drop=True)
    print(product_not_in_order)
    return product_Feature

def returnNotMatches(a, b):
    a = set(a)
    b = set(b)
    return list(b-a),list(a-b)

if __name__ =='__main__':
    category_to_be_considered = [101,103,301,401,602,603,701,1009,1102,1106,1333]
    file_path = "D:\\RecommendationEngine\\CatalogData_Lates\\CatalogData\\Clothing\\product_attributes.csv"
    file_path_orders = "D:\\RecommendationEngine\\Orders\\Orders\\*.csv"
    product_Feature = preProcessingProductFeature(file_path,file_path_orders,category_to_be_considered)
    with open("Product_Feature_Sparse_Matrix.pickle",'wb') as handle:
        pickle.dump(product_Feature, handle, protocol=pickle.HIGHEST_PROTOCOL)