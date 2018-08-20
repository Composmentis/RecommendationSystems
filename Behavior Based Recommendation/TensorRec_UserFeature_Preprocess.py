import tensorrec
import tensorflow as tf
import pandas as pd
import glob as gb
import pickle
from scipy import sparse
from sklearn import preprocessing

def pivotingData(dataFrame,indexColumn,columnsColumn,valuesColumn):
    return dataFrame.pivot_table(index=indexColumn, columns=columnsColumn, values=valuesColumn)

def splitDataFrameIntoSmaller(df, chunkSize = 3000):
    listOfDf = list()
    numberChunks = len(df) // chunkSize + 1
    for i in range(numberChunks):
        listOfDf.append(df[i*chunkSize:(i+1)*chunkSize])
    return listOfDf

#def populateUserData(user_does_not_have_city,ZIPCODE,CITY):
# def missingUsers(user_Feature):
#     file_path = "D:\\RecommendationEngine\\Orders\\Orders*.csv"
#     user_from_order = []
#     for DataFrame in gb.glob(file_path):
#         #print(DataFrame)
#         Order_History = pd.read_csv(DataFrame, encoding="ISO-8859-1", header=0)
#         Order_History = Order_History['MEMBER_ID'].tolist()
#         user_from_order = user_from_order + Order_History
#     user_from_order = set(user_from_order)
#     user_from_order = list(user_from_order)
#     #print(len(user_from_order))
#     #print(len(list(set(user_Feature['MEMBER_ID'].tolist()))))
#     user_Feature = (list(set(user_Feature['MEMBER_ID'].tolist())))
#     user_does_not_have_city, user_not_in_order = returnNotMatches(user_Feature, user_from_order)
#     return user_not_in_order, user_does_not_have_city

def returnNotMatches(a, b):
    a = set(a)
    b = set(b)
    return list(b-a),list(a-b)

def preProcessingUserFeature(file_path):
    # file_path = "D:\\RecommendationEngine\\Users\\Users-Orders\\Users-Orders.csv"
    user_Feature = pd.read_csv(file_path, encoding="ISO-8859-1", header=0)
    # user_not_in_order, user_does_not_have_city = missingUsers(user_Feature)
    #print(len(user_not_in_order))
    # for user in user_not_in_order:
    #     print("user not in order")
    #     user_Feature = user_Feature[user_Feature.MEMBER_ID !=user]
    # #populateUserData(user_does_not_have_city,user_Feature.ZIPCODE.unique(),user_Feature.CITY.unique())
    user_Feature = user_Feature.groupby(["MEMBER_ID"])["CITY",'ZIPCODE'].first().reset_index()
    user_Feature['value']=1
    user_Feature_List = splitDataFrameIntoSmaller(user_Feature)

    # user_Feature_temp_List = []
    # for user in user_Feature_List:
    #     #print("city")
    #     temp =pivotingData(user, "MEMBER_ID", "CITY", "value")
    #     user_Feature_temp_List.append(temp.reset_index())
    # user_Feature_1 = pd.concat(user_Feature_temp_List, axis=0, ignore_index=True)
    # #print("user_Feature_1 is done")
    user_Feature_temp_List = []
    for user in user_Feature_List:
        #print("zipcode")
        temp =pivotingData(user, "MEMBER_ID", "ZIPCODE", "value")
        user_Feature_temp_List.append(temp.reset_index())
    user_Feature = pd.concat(user_Feature_temp_List, axis=0, ignore_index=True)
    #print("user_Feature_2 is done")
    # user_Feature = pd.merge(user_Feature_1,user_Feature_2,on="MEMBER_ID",how="outer")
    del user_Feature_List,user_Feature_temp_List

    user_Feature = user_Feature.fillna(0)
    user_Feature=user_Feature.rename(columns ={"MEMBER_ID":"Customer_Id"})
    #Deleting user not considered for recommendation
    with open("Customer_Products.pickle", "rb") as uf:
        user_for_recommendation = pickle.load(uf)
    user_for_recommendation = user_for_recommendation['Customer']
    print(type(user_Feature))
    temp =user_Feature['Customer_Id'].tolist()
    usertobedeleted,temp=returnNotMatches(user_for_recommendation,temp)
    user_Feature = user_Feature.drop(
        user_Feature[user_Feature.Customer_Id.isin(usertobedeleted)].index.tolist())
    print(returnNotMatches(user_Feature['Customer_Id'].tolist(), user_for_recommendation))
    user_Feature=user_Feature.set_index("Customer_Id")
    del user_Feature.index.name
    user_Feature.to_csv("user_Feature.csv")
    user_Feature = sparse.coo_matrix(user_Feature)
    #print(user_Feature)
    return user_Feature

if __name__ =='__main__':
    file_path = "D:\\RecommendationEngine\\Users\\Users-Orders\\Users-Orders.csv"
    user_Feature = preProcessingUserFeature(file_path)
    with open("User_Feature_Sparse_Matrix.pickle",'wb') as handle:
        pickle.dump(user_Feature, handle, protocol=pickle.HIGHEST_PROTOCOL)
    interactions, user_features, item_features = tensorrec.util.generate_dummy_data(num_users=122364,
                                                                                    num_items=150,
                                                                                    interaction_density=.05)
    with open("User_Features_Sparse_Matrix.pickle",'wb') as handle:
        pickle.dump(user_features, handle, protocol=pickle.HIGHEST_PROTOCOL)

