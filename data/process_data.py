import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load dataframe from filepaths
    INPUT
    messages_filepath - str, link to file
    categories_filepath - str, link to file
    OUTPUT
    df - pandas DataFrame
    """
    messages_filepath = 'disaster_messages.csv'
    categories_filepath = 'disaster_categories.csv'
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    '''
    Clean the data
    
    INPUT - a data frame with merged messages and categories
    
    OUTPUT
    df that has following charachteristics:
    1. The column categories is splited into 36 different categories
    2. The columns are renamed acordingly
    3. Values for each column are int without strings
    4. Has no column 'categories'
    5. Has id, messages, original, genre and 36 columns with feautures
    6. Has no duplicates
    
    '''
    
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.str.split('-').apply(lambda x: x[0])
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').apply(lambda x: x[1])
    
    #convert column from string to numeric
    categories = categories.astype(int) 
    
    df = df.drop(columns='categories')
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    '''
    Saves df into database
    
    '''
    database_filename = 'sqlite:///DisasterResponse.db'
    engine = create_engine(database_filename)
    df.to_sql('df', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
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