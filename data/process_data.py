import sys
import pandas as pd
import numpy as np
import sqlalchemy as db
from langdetect import detect


def load_data(messages_filepath, categories_filepath):
    '''
    USAGE 
           load the data that we'll use to train the model
    INPUT
           messages_filepath: csv file with the messages
           categories_filepath: csv file with the categories of the messages           
    OUTPUT
           df: dataframe with the information of the two files.           
    '''
    # read messages file
    messages_df = pd.read_csv(messages_filepath)
    # read categories file
    categories_df = pd.read_csv(categories_filepath)
    # merge the two files
    df = messages_df.merge(categories_df, on=['id'])
    return df


# detect the language of the message.
def language_message(df):
    '''
    USAGE 
           uses langdetect to infere the language of the original message when
           it is available
    INPUT
           df: dataframe with the messages, original messages, and all the data
               extracted from the original files.           
    OUTPUT
           df: the same dataframe with an additional column with the language           
    '''
    langdet = []
    for index, row in df.iterrows():

        try:
            language = detect(row.original)
            print(language)
        except e:
            language = "unknown"
            print(e)
        langdet.append(language)
    

    lang = pd.Series(langdet)
    df['lang'] = lang.values
    return df    

def clean_data(df):
    '''
    USAGE 
           clean the data. 
             - Save the information of categories in a column for
           each category. 
             - Drop duplicates.
             - Drop categories without any message.
             - Drop rows with related value = 2.
             - Drop rows with messages shorter than 27 characters.
    INPUT
           df: dataframe with the messages, original messages, and all the data
               extracted from the original files.           
    OUTPUT
           df: the same dataframe with the cleaned data.           
    '''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat = ';', expand = True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.str.slice(start = 0, stop = -2)
    categories.columns = category_colnames
    
    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(start = -1)
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    df = df.drop(columns = ['categories'])
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    # drop column child_alone. We don't have any messages of this label
    df = df.drop(columns = ['child_alone'])
    
    # drop all the rows where related = 2.
    df = df[~(df['related'] == 2)]
    
    # drop the rows where the length of the message is samaller than 27 chars
    df['len'] = df.message.apply(len)
    df = df[df['len'] >= 27]
    
    return df

def save_data(df, database_filename):
    '''
    USAGE 
           Save the dataframe data into a SQLite database
    INPUT
           df: data we want to save.  
           database_filename: file name of the databese where we wanto to save
             the data.                      
    '''
    # open a connection with the database
    engine = db.create_engine('sqlite:///' + database_filename)
    connection = engine.connect()
    # save the dataframe df into the table messages
    df.to_sql('messages', engine, index=False, if_exists='replace')
    #close the connection with database
    return connection.close()


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print('Detect language...')
        df = language_message(df)
        
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