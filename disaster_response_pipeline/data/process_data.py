import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Loads input data from messages_filepath and categories_filepath to resulting dataset
    
    Inputs:
        messages_filepath: string, path to CSV 'messages' file
        categories_filepath: string, path to CSV 'categories' file
    Output:
        df: dataframe, result from a merge/left join operation between messages and categories, using the 'id' column as key
    '''
    # loading CSV datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merging in order to obtain df
    df = messages.merge(categories, on = ['id'], how = 'left')
    
    return df


def clean_data(df):
    '''
    Cleans the input dataframe df, by transforming its categorical columns into 0s and 1s, and removing duplicates
    
    Input:
        df: dataframe, input dataset which will be cleaned/treated
    Output:
        clean_df: dataframe, resulting dataset after cleaning operations
    '''
    # splitting column values using ';'
    categories = df.categories.str.split(';', expand = True)
    
    # selecting the first row of the categories dataframe and using it as reference to extract new column names
    row = categories.loc[0, ]
    category_colnames = [i[0:-2] for i in list(row)]
    categories.columns = category_colnames
    
    # converting categories values to 0 and 1:
    for column in categories:
        # setting each value to be the last character of the string, and then converting to integer
        categories[column] = categories[column].astype(str).str[-1].astype(int)

    # dropping the original categories column from df, and then concatenating df and categories
    df.drop(['categories'], axis = 1, inplace = True)
    df = pd.concat([df, categories], axis = 1)
    
    # removing duplicates:
    clean_df = df.drop_duplicates()
    
    return clean_df


def save_data(df, database_filename):
    '''
    Saves dataframe df in SQlAlchemy database file, after database_filename
    
    Input:
        df: dataframe, dataset to be saved in SQLAlchemy database file
        database_filename: string, SQLAchemy database file name
    Output:
        None
    '''
    # setting up engine with database_filname
    engine = create_engine('sqlite:///{}'.format(database_filename))
    
    # saving df in the database as MessagesCategories
    df.to_sql('disaster', engine, index=False, if_exists = 'replace')


def main():
    '''
    Executes code, using previously defined functions - reads datasets, treats them, then saves them to SQLAlchemy database file
    
    Input: 
        None
    Output: 
        None
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {} \n    with {} rows'.format(database_filepath, str(df.shape[0])))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()