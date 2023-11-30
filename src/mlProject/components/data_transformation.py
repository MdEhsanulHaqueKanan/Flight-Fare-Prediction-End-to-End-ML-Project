import os
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from mlProject.entity.config_entity import DataTransformationConfig
#New Code Added Start
from sklearn.ensemble import ExtraTreesRegressor
#New Code Added End


class DataTransformation:

    # New Function Added
    # https://github.com/yash1314/Flight-Price-Prediction/blob/main/src/utils.py
    def convert_to_minutes(self, duration):
        try:
            hours, minute = 0, 0
            for i in duration.split():
                if 'h' in i:
                    hours = int(i[:-1])
                elif 'm' in i:
                    minute = int(i[:-1])
            return hours * 60 + minute
        except :
            return None 

    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up

    # New Code Added Start
    def initiate_data_transformation(self):
        ## reading the data
        # df = pd.read_csv(self.config.data_path)
        # New Line
        df = pd.read_excel(self.config.data_path)

        # I added this line to see the data path
        logger.info(f' data path: \n{self.config.data_path}')
        logger.info('Read data completed')
        logger.info(f'df dataframe head: \n{df.head().to_string()}')

        ## dropping null values
        df.dropna(inplace = True)

        ## Date of journey column transformation
        df['journey_date'] = pd.to_datetime(df['Date_of_Journey'], format ="%d/%m/%Y").dt.day
        df['journey_month'] = pd.to_datetime(df['Date_of_Journey'], format ="%d/%m/%Y").dt.month

        ## encoding total stops.
        df.replace({'Total_Stops': {'non-stop' : 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}}, inplace = True)

        ## ecoding airline, source, and destination
        df_airline = pd.get_dummies(df['Airline'], dtype=int)
        df_source = pd.get_dummies(df['Source'],  dtype=int)
        df_dest = pd.get_dummies(df['Destination'], dtype=int)

        ## dropping first columns of each categorical variables.
        df_airline.drop('Trujet', axis = 1, inplace = True)
        df_source.drop('Banglore', axis = 1, inplace = True)
        df_dest.drop('Banglore', axis = 1, inplace = True)

        df = pd.concat([df, df_airline, df_source, df_dest], axis = 1)
       
        ## handling duration column
        # df['duration'] = df['Duration'].apply(convert_to_minutes)
        # New Line Added
        df['duration'] = df['Duration'].apply(self.convert_to_minutes)
        upper_time_limit = df.duration.mean() + 1.5 * df.duration.std()
        df['duration'] = df['duration'].clip(upper = upper_time_limit)

        ## encodign duration column
        bins = [0, 120, 360, 1440]  # custom bin intervals for 'Short,' 'Medium,' and 'Long'
        labels = ['Short', 'Medium', 'Long'] # creating labels for encoding

        df['duration'] = pd.cut(df['duration'], bins=bins, labels=labels)
        df.replace({'duration': {'Short':1, 'Medium':2, 'Long': 3}}, inplace = True)
        
        ## dropping the columns
        cols_to_drop = ['Airline', 'Date_of_Journey', 'Source', 'Destination', 'Route', 'Dep_Time', 'Arrival_Time', 'Duration', 'Additional_Info', 'Delhi', 'Kolkata']
        df.drop(cols_to_drop, axis = 1, inplace = True)

        logger.info('df data transformation completed')
        logger.info(f' transformed df data head: \n{df.head().to_string()}')
        

        # df.to_csv(self.data_transformation_config.transformed_data_file_path, index = False, header= True)
        # New Line
        df.to_excel(self.config.data_path, index = False, header= True)
        # df.to_excel(self.config.transformed_data_file_path, index = False, header= True)
        # df.to_excel(self.data_transformation_config.transformed_data_file_path, index = False, header= True)
        logger.info("transformed data is stored")
        df.head(1)
        ## splitting the data into training and target data
        X = df.drop('Price', axis = 1)
        y = df['Price']
        
        ## accessing the feature importance.
        select = ExtraTreesRegressor()
        select.fit(X, y)

        # plt.figure(figsize=(12, 8))
        # fig_importances = pd.Series(select.feature_importances_, index=X.columns)
        # fig_importances.nlargest(20).plot(kind='barh')
    
        # ## specify the path to the "visuals" folder using os.path.join
        # visuals_folder = 'visuals'
        # if not os.path.exists(visuals_folder):
        #     os.makedirs(visuals_folder)

        # ## save the plot in the visuals folder
        # plt.savefig(os.path.join(visuals_folder, 'feature_importance_plot.png'))
        # logger.info('feature imp figure saving is successful')

        ## further Splitting the data.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, shuffle = True) 
        # New Line Added After Completing Data Transformation Sucessfully
        y_train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        y_test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)
        # New Line Ended After Completing Data Transformation Sucessfully        
        logger.info('final splitting the data is successful')
        # logger.info(y_train.shape)
        # logger.info(y_test.shape)        
        

        ## returning splitted data and data_path.
        return (
            X_train, 
            X_test, 
            y_train, 
            y_test,
            self.config.data_path
            # self.data_transformation_config.transformed_data_file_path
        )    

           


# class DataTransformation:
#     def __init__(self, config: DataTransformationConfig):
#         self.config = config

    
#     ## Note: You can add different data transformation techniques such as Scaler, PCA and all
#     #You can perform all kinds of EDA in ML cycle here before passing this data to the model

#     # I am only adding train_test_spliting cz this data is already cleaned up


#     def train_test_spliting(self):
#         data = pd.read_excel(self.config.data_path)

#         # Split the data into training and test sets. (0.75, 0.25) split.
#         train, test = train_test_split(data)

#         train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
#         test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

#         logger.info("Splited data into training and test sets")
#         logger.info(train.shape)
#         logger.info(test.shape)

#         print(train.shape)
#         print(test.shape)
        




# class DataTransformation:
#     def __init__(self, config: DataTransformationConfig):
#         self.config = config

    
#     ## Note: You can add different data transformation techniques such as Scaler, PCA and all
#     #You can perform all kinds of EDA in ML cycle here before passing this data to the model

#     # I am only adding train_test_spliting cz this data is already cleaned up

#     # New Code Added Start
#     def initiate_data_transformation(self):
#         ## reading the data
#         df = pd.read_csv(self.config.data_path)

#         logger.info('Read data completed')
#         logger.info(f'df dataframe head: \n{df.head().to_string()}')

#         ## dropping null values
#         df.dropna(inplace = True)

#         ## Date of journey column transformation
#         df['journey_date'] = pd.to_datetime(df['Date_of_Journey'], format ="%d/%m/%Y").dt.day
#         df['journey_month'] = pd.to_datetime(df['Date_of_Journey'], format ="%d/%m/%Y").dt.month

#         ## encoding total stops.
#         df.replace({'Total_Stops': {'non-stop' : 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}}, inplace = True)

#         ## ecoding airline, source, and destination
#         df_airline = pd.get_dummies(df['Airline'], dtype=int)
#         df_source = pd.get_dummies(df['Source'],  dtype=int)
#         df_dest = pd.get_dummies(df['Destination'], dtype=int)

#         ## dropping first columns of each categorical variables.
#         df_airline.drop('Trujet', axis = 1, inplace = True)
#         df_source.drop('Banglore', axis = 1, inplace = True)
#         df_dest.drop('Banglore', axis = 1, inplace = True)

#         df = pd.concat([df, df_airline, df_source, df_dest], axis = 1)

#         # New Function Added
#         # https://github.com/yash1314/Flight-Price-Prediction/blob/main/src/utils.py
#         def convert_to_minutes(duration):
#             try:
#                 hours, minute = 0, 0
#                 for i in duration.split():
#                     if 'h' in i:
#                         hours = int(i[:-1])
#                     elif 'm' in i:
#                         minute = int(i[:-1])
#                 return hours * 60 + minute
#             except :
#                 return None        

#         ## handling duration column
#         df['duration'] = df['Duration'].apply(convert_to_minutes)
#         upper_time_limit = df.duration.mean() + 1.5 * df.duration.std()
#         df['duration'] = df['duration'].clip(upper = upper_time_limit)

#         ## encodign duration column
#         bins = [0, 120, 360, 1440]  # custom bin intervals for 'Short,' 'Medium,' and 'Long'
#         labels = ['Short', 'Medium', 'Long'] # creating labels for encoding

#         df['duration'] = pd.cut(df['duration'], bins=bins, labels=labels)
#         df.replace({'duration': {'Short':1, 'Medium':2, 'Long': 3}}, inplace = True)
        
#         ## dropping the columns
#         cols_to_drop = cols_to_drop = ['Airline', 'Date_of_Journey', 'Source', 'Destination', 'Route', 'Dep_Time', 'Arrival_Time', 'Duration', 'Additional_Info', 'Delhi', 'Kolkata']

#         df.drop(cols_to_drop, axis = 1, inplace = True)

#         logger.info('df data transformation completed')
#         logger.info(f' transformed df data head: \n{df.head().to_string()}')

#         df.to_csv(self.data_transformation_config.transformed_data_file_path, index = False, header= True)
#         logger.info("transformed data is stored")
#         df.head(1)
#         ## splitting the data into training and target data
#         X = df.drop('Price', axis = 1)
#         y = df['Price']
        
#         ## accessing the feature importance.
#         select = ExtraTreesRegressor()
#         select.fit(X, y)

#         plt.figure(figsize=(12, 8))
#         fig_importances = pd.Series(select.feature_importances_, index=X.columns)
#         fig_importances.nlargest(20).plot(kind='barh')
    
#         ## specify the path to the "visuals" folder using os.path.join
#         visuals_folder = 'visuals'
#         if not os.path.exists(visuals_folder):
#             os.makedirs(visuals_folder)

#         ## save the plot in the visuals folder
#         plt.savefig(os.path.join(visuals_folder, 'feature_importance_plot.png'))
#         logger.info('feature imp figure saving is successful')

#         ## further Splitting the data.
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, shuffle = True) 
#         logger.info('final splitting the data is successful')
        
#         ## returning splitted data and data_path.
#         return (
#             X_train, 
#             X_test, 
#             y_train, 
#             y_test,
#             self.data_transformation_config.transformed_data_file_path
#         )    
#     # New Code Added End

#     # def train_test_spliting(self):
#     #     data = pd.read_excel(self.config.data_path)

#     #     # Split the data into training and test sets. (0.75, 0.25) split.
#     #     train, test = train_test_split(data)

#     #     train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
#     #     test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

#     #     logger.info("Splited data into training and test sets")
#     #     logger.info(train.shape)
#     #     logger.info(test.shape)

#     #     print(train.shape)
#     #     print(test.shape)