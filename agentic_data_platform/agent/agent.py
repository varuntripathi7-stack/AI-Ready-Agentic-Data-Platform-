#!/usr/bin/env python3
"""
AI Agent for E-commerce Data Platform
Uses Ollama with Mistral model to answer natural language questions.
Executes SQL queries on Gold layer Delta tables.
"""

import json # For any JSON handling if needed
import re # For parsing SQL-like queries



# Import typing utilities for type hints.
# - List: Represents a list of specific data types (e.g., List[int])
# - Dict: Represents a dictionary with defined key-value types (e.g., Dict[str, int])
# - Any: Allows any data type when the type is unknown or flexible
# - Optional: Indicates a value can be of a specific type or None
from typing import List, Dict, Any, Optional  



# Import datetime class to work with date and time operations
# (e.g., timestamps, logging execution time, time calculations)
from datetime import datetime

# LangChain imports for working with LLMs and prompts
# LangChain is a Python (and JavaScript) framework that helps you build applications using Large Language Models (LLMs) like GPT, Llama, Mistral, etc.:
# LangChain = A toolkit to build AI-powered apps easily

# Import Ollama LLM from LangChain - Ollama is a local LLM server that can run models like Mistral and Phi on your machine, providing fast and private access to powerful language models.
try: 
    from langchain_ollama import OllamaLLM  # Try to import from langchain_ollama first (newer versions)
except ImportError:
    from langchain_community.llms import Ollama as OllamaLLM # Fallback to older import path if the first one fails



# For reading Delta tables
import pandas as pd # For data manipulation and analysis (e.g., reading parquet files, handling DataFrames)
import glob # For file pattern matching (e.g., to find all parquet files in a directory)






# LLM Configuration

# OLLAMA_MODEL:
#   Specifies the local Ollama model to use for generating responses.
#   "phi" is selected because it is lightweight and faster than larger models
#   like Mistral, making it suitable for local development and quick inference.
#
# OLLAMA_BASE_URL:
#   Defines the endpoint where the Ollama server is running.
#   By default, Ollama serves models locally at http://localhost:11434.
#   This allows the application to send prompts to the model via API calls.
OLLAMA_MODEL = "TinyLlama"  # Specifying the model to use with Ollama (Mistral as required by the assignment)
OLLAMA_BASE_URL = "http://localhost:11434" # Default Ollama endpoint



# Import the built-in 'os' module with an alias (os_module)
# to avoid potential naming conflicts with other variables named 'os'.
import os as os_module 


# Determine the project’s base directory dynamically.
# Step-by-step:
# 1. __file__ → Gets the current file's path.
# 2. abspath(__file__) → Converts it to an absolute path.
# 3. dirname(...) → Moves one level up (parent directory).
# 4. dirname(...) again → Moves one more level up.
#
# Result:
# BASE_PATH points to the root directory of the project.
# This ensures file paths work correctly regardless of where the script is executed from.
BASE_PATH = os_module.path.dirname(os_module.path.dirname(os_module.path.abspath(__file__))) # Get the base path of the project by going up two levels from the current file's location

GOLD_PATHS = {
    "revenue_per_hour": os_module.path.join(BASE_PATH, "data/gold/revenue_per_hour"), 
    "active_users_per_hour": os_module.path.join(BASE_PATH, "data/gold/active_users_per_hour"),
    "conversion_rate": os_module.path.join(BASE_PATH, "data/gold/conversion_rate")
} # Define paths to the Gold layer Delta tables for revenue, active users, and conversion rate


# Define path to the features table (user_features) in the features layer
FEATURES_PATH = os_module.path.join(BASE_PATH, "data/features/user_features") 





# DataQueryEngine class is responsible for loading Delta Lake tables as pandas DataFrames and executing SQL-like queries on them.
# It provides methods to load tables, get schema information, execute queries, and summarize data for LLM context.
# The class abstracts away the complexities of reading Delta tables and allows the AI agent to interact with the data in a more intuitive way.
# Key functionalities include:
# - Loading Delta tables from specified paths and storing them in a dictionary for easy access.
# - Providing schema information for all loaded tables.
# - Executing simple SQL-like queries (SELECT, WHERE, ORDER BY, LIMIT) on the loaded DataFrames.
# - Summarizing data for use in LLM prompts, giving the AI agent context about the available data when answering questions.         
class DataQueryEngine:
    """
    Engine for querying Delta Lake tables using pandas SQL.
    Loads tables from the Gold layer and features, and allows executing SQL-like queries.

    Methods:
    - __init__: Initializes the engine and loads tables.
    - _load_delta_table: Helper method to load a Delta table as a pandas DataFrame.
    - _load_tables: Loads all specified tables into memory.
    - get_table_schemas: Returns schema information for all loaded tables.
    - execute_query: Executes a SQL-like query on the loaded tables and returns results.
    - _execute_select: Helper method to execute SELECT queries using pandas.
    - get_summary_stats: Provides summary statistics for a specified table.
    - get_data_context: Summarizes all loaded data for use in LLM context.  
    """
    
    def __init__(self): #Initialize the DataQueryEngine and load tables into memory
        self.tables: Dict[str, pd.DataFrame] = {} #  define a dictionary to hold the loaded tables, where keys are table names and values are pandas DataFrames
        self._load_tables() # Load all tables from the specified paths into memory when the engine is initialized
    


    # _load_delta_table is a helper method that takes a file path as input and attempts to load all parquet files 
    # in that directory into a single pandas DataFrame. 
    # It uses glob to find all parquet files, reads them into DataFrames, and concatenates them. 
    # If successful, it returns the combined DataFrame; 
    # if no parquet files are found or an error occurs, it returns None and prints a warning message.
    def _load_delta_table(self, path: str) -> Optional[pd.DataFrame]: #
        """
        Load a Delta table as pandas DataFrame.
        Args: path (str): The file path to the Delta table (parquet files).
        Returns: Optional[pd.DataFrame]: A pandas DataFrame containing the table data, or None if loading fails.

        """
        try:
            # Prefer delta-rs to read only active Delta files (avoids stale parquet)
            try:
                from deltalake import DeltaTable
                dt = DeltaTable(path)
                return dt.to_pandas()
            except Exception:
                pass
            # Fallback to reading parquet files directly
            parquet_files = glob.glob(f"{path}/*.parquet")
            if parquet_files:
                df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
                return df
            else:
                print(f"  ⚠ No parquet files found in {path}")
                return None
        except Exception as e:
            print(f"  ⚠ Error loading {path}: {e}")
            return None
    


    # _load_tables is a helper method that iterates through the GOLD_PATHS dictionary, loading each specified Delta table using the _load_delta_table method.
    # It stores the loaded DataFrames in the self.tables dictionary with the table name as the key.
    # # It also loads the user_features table from the FEATURES_PATH. 
    # The method prints the status of each table loading operation, 
    # including the number of rows loaded or any warnings if loading fails. 
    def _load_tables(self):
        """
        Load all available tables.
          - Iterates through the GOLD_PATHS to load each Gold layer table.
          - Also loads the user_features table from the features layer. 
          - Stores loaded tables in the self.tables dictionary for easy access.

        """
        print("Loading Gold layer tables...")
        
        for table_name, path in GOLD_PATHS.items():
            df = self._load_delta_table(path)
            if df is not None:
                self.tables[table_name] = df
                print(f"  ✓ Loaded {table_name}: {len(df)} rows")
        
        # Also load features table
        df = self._load_delta_table(FEATURES_PATH)
        if df is not None:
            self.tables["user_features"] = df
            print(f"  ✓ Loaded user_features: {len(df)} rows")
    



    # get_table_schemas returns a string containing schema information for all loaded tables.
    # It iterates through self.tables, extracting column names and their data types for each table.
    def get_table_schemas(self) -> str:
        """
        Get schema information for all tables.

        Returns a formatted string with table names, columns, data types, and row counts. 
          - If no tables are loaded, it returns a message indicating that the Spark pipeline should be run first. 

        
        """
        schemas = []
        for table_name, df in self.tables.items():
            columns = ", ".join([f"{col} ({df[col].dtype})" for col in df.columns])
            schemas.append(f"Table: {table_name}\n  Columns: {columns}\n  Rows: {len(df)}")
        return "\n\n".join(schemas) if schemas else "No tables loaded yet. Run the Spark pipeline first."




    # execute_query takes a SQL-like query as input and attempts to execute it against the loaded tables.
    # It supports basic SELECT queries with optional WHERE, ORDER BY, and LIMIT clauses.
    def execute_query(self, query: str) -> str:
        """
        Execute a pandas query on the available tables.

         - Parses the query to identify the table and any conditions.
         - Supports basic SELECT queries with optional WHERE, ORDER BY, and LIMIT clauses. 
         - If the query is a DESCRIBE or SHOW TABLES command, it returns the schema information. 
         - If the query references a table directly, it returns the first few rows of that table.
         - If the query cannot be parsed, it returns an error message with available tables. 
         - Catches and returns any exceptions that occur during query execution.

       Args: query (str): The SQL-like query to execute.
       Returns: str: The result of the query execution or an error message.

        """
        try:
            query = query.strip() # Remove leading and trailing whitespace from the query for cleaner parsing
            
            if query.lower().startswith("select"): #If the query starts with "SELECT", we will attempt to execute it as a SQL-like query using pandas
                return self._execute_select(query) #Call the helper method to execute the SELECT query and return the result
            elif query.lower().startswith("describe") or query.lower().startswith("show"): #If the query starts with "DESCRIBE" or "SHOW", we will return the schema information for all loaded tables
                return self.get_table_schemas() #Call the method to get table schemas and return it as the result
            else: # For any other query, we will check if it references a table directly and return the first few rows of that table
                for table_name in self.tables: # Iterate through the loaded tables to see if the query references any of them   
                    if table_name.lower() in query.lower(): # If the table name is found in the query (case-insensitive match), we will return the first few rows of that table
                        df = self.tables[table_name] # Get the DataFrame for the matched table
                        return f"Table {table_name}:\n{df.head(10).to_string()}" # Return the first 10 rows of the matched table as a string
                
                return f"Could not parse query. Available tables: {list(self.tables.keys())}"
        except Exception as e:
            return f"Query error: {str(e)}" #Catch and return any exceptions that occur during query execution as an error message
    



    # execute_select is a helper method that executes SELECT-like queries using pandas.
    # It parses the query to identify the table name, optional WHERE conditions, ORDER BY clauses, and LIMIT.
    # It uses regular expressions to extract these components from the query string and applies them to the corresponding DataFrame.
    # The method returns the resulting DataFrame as a string. If any errors occur during parsing or execution, it returns an error message.
    # This method allows the AI agent to execute simple SQL-like queries on the loaded DataFrames without needing a full SQL engine, enabling it to answer questions based on the data effectively.
    def _execute_select(self, query: str) -> str:
        """
        Execute a SELECT-like query using pandas.
         - Parses the query to identify the table, optional WHERE conditions, ORDER BY clauses, and LIMIT. 
         - Uses regular expressions to extract these components from the query string. 
         - Applies the conditions and sorting to the corresponding DataFrame. - Returns the resulting DataFrame as a string. 
         - If any errors occur during parsing or execution, it returns an error message. Args: query (str): 
         The SQL-like SELECT query to execute. Returns: str: The result of the query execution or an error message.
        
        """
        try:
            from_match = re.search(r'from\s+(\w+)', query, re.IGNORECASE) # Use regular expression to find the table name after the "FROM" keyword in the query (case-insensitive match)
            if not from_match:
                return "Could not identify table in query"
            
            table_name = from_match.group(1) # Extract the table name from the regex match group
            
            matched_table = None
            for t in self.tables: # Iterate through the loaded tables to find a match for the table name in the query (case-insensitive match or partial match)
                if t.lower() == table_name.lower() or table_name.lower() in t.lower(): # If the table name in the query matches a loaded table name (case-insensitive) or is a substring of a loaded table name, we consider it a match
                    matched_table = t #
                    break
            
            if not matched_table: # If no matching table is found, return an error message with the available tables
                return f"Table '{table_name}' not found. Available: {list(self.tables.keys())}" # 
            
            df = self.tables[matched_table].copy() # Get a copy of the matched table's DataFrame to work with for query execution
            
            where_match = re.search(r'where\s+(.+?)(?:order|group|limit|$)', query, re.IGNORECASE) # Use regular expression to find the WHERE clause and its conditions in the query (case-insensitive match). It captures everything after "WHERE" until it hits "ORDER", "GROUP", "LIMIT", or the end of the string.
            if where_match: # If a WHERE clause is found in the query, we will attempt to apply the conditions to filter the DataFrame accordingly
                condition = where_match.group(1).strip() # Extract the conditions from the regex match group and remove any leading/trailing whitespace
                condition = condition.replace("<>", "!=") # Replace SQL-style <> with !=
                condition = re.sub(r'(?<![<>!])=(?!=)', '==', condition) # Replace single = with == while preserving >=, <=, !=, and ==
                try:
                    df = df.query(condition) # Use the pandas query method to filter the DataFrame based on the extracted conditions. If the condition is valid, it will return a new DataFrame with only the rows that satisfy the condition.
                except: # If there is an error in the condition (e.g., syntax error, invalid column name), we will ignore the WHERE clause and proceed without filtering the DataFrame. This allows us to still return results even if the condition is not perfectly formatted.
                    pass
             
            order_match = re.search(r'order\s+by\s+(\w+)\s*(asc|desc)?', query, re.IGNORECASE) # Use regular expression to find the ORDER BY clause in the query (case-insensitive match). It captures the column name to sort by and an optional sorting direction (ASC or DESC).
            if order_match: # If an ORDER BY clause is found in the query, we will attempt to sort the DataFrame based on the specified column and sorting direction
                col = order_match.group(1) # Extract the column name to sort by from the regex match group
                ascending = order_match.group(2) is None or order_match.group(2).lower() == 'asc' # Determine the sorting direction based on the optional second regex group. If it is not specified or is "ASC", we will sort in ascending order; if it is "DESC", we will sort in descending order.
                if col in df.columns:  # Check if the specified column for sorting exists in the DataFrame. If it does, we will sort the DataFrame based on that column and the determined sorting direction.
                    df = df.sort_values(col, ascending=ascending) # Use the pandas sort_values method to sort the DataFrame by the specified column and sorting direction. If the column does not exist, we will skip sorting and proceed with the original order of the DataFrame.
            
            limit_match = re.search(r'limit\s+(\d+)', query, re.IGNORECASE) # Use regular expression to find the LIMIT clause in the query (case-insensitive match). It captures the number of rows to limit the result to.     
            if limit_match: # If a LIMIT clause is found in the query, we will attempt to limit the number of rows in the resulting DataFrame to the specified number. This allows us to return only a subset of the results if the query includes a LIMIT clause.
                limit = int(limit_match.group(1)) # Extract the number of rows to limit to from the regex match group and convert it to an integer
                df = df.head(limit) # Use the pandas head method to limit the DataFrame to the specified number of rows. If no LIMIT clause is found, we will return the first 10 rows of the resulting DataFrame by default to avoid overwhelming the output with too much data.
            else: # If no LIMIT clause is found in the query, we will return the first 10 rows of the resulting DataFrame by default to provide a manageable amount of data in the output. This ensures that even if the query does not specify a limit, we won't return an excessively large result set that could be difficult to read or process.
                df = df.head(10) # Return the first 10 rows of the resulting DataFrame as a string. This provides a snapshot of the query results while keeping the output concise and readable. If the resulting DataFrame is empty, it will return an empty string or indicate that no results were found based on how pandas handles empty DataFrames when converted to strings.
            
            return df.to_string() # Convert the resulting DataFrame to a string format for display. This allows us to return the query results in a readable format that can be easily printed or included in responses from the AI agent. If the DataFrame is large, it will be truncated based on the earlier LIMIT clause or default head(10) to ensure the output remains manageable.
        except Exception as e: # Catch and return any exceptions that occur during the parsing or execution of the query as an error message. This helps to provide feedback on what went wrong if the query is not properly formatted or if there are issues with the data.
            return f"Query execution error: {str(e)}" # Return an error message indicating that there was an issue executing the query, along with the specific error message from the exception. This allows users to understand what went wrong and potentially adjust their query accordingly.
    


    # get_summary_stats provides summary statistics for a specified table. 
    # It checks if the table exists in the loaded tables, and if it does, it uses the pandas describe() method 
    # to generate summary statistics for that table. 
    # The resulting statistics are returned as a formatted string. 
    # If the specified table is not found, it returns an error message indicating that the table was not found. 
    # This method can be used to give the AI agent a quick overview of the data in a specific table when answering questions.
    def get_summary_stats(self, table_name: str) -> str: 
        """
        Get summary statistics for a table.

        Args: table_name (str): The name of the table to summarize.
        Returns: str: A summary of the table's statistics or an error message if the table is not found.
         - Checks if the specified table exists in the loaded tables. 
         - If the table exists, it uses the pandas describe() method to generate summary statistics for that table and returns it as a formatted string. 
         - If the specified table is not found in the loaded tables, it returns an error message indicating that the table was not found.
         - This method can be used to provide the AI agent with a quick overview of the data in a specific table when answering questions, giving insights into the distribution of values, counts, means, standard deviations, and other relevant statistics for the columns in that table. 
         - If the table contains non-numeric columns, the describe() method will provide summary statistics for those as well, such as counts, unique values, top values, and frequency for categorical data. 
         - This allows the AI agent to have a better understanding of the data when formulating responses to user questions based on the contents of the specified table.           

        """
        if table_name not in self.tables: # Check if the specified table name exists in the loaded tables. If it does not exist, we will return an error message indicating that the table was not found.
            return f"Table '{table_name}' not found" # Return an error message indicating that the specified table was not found in the loaded tables. This helps to provide feedback to the user if they request summary statistics for a table that does not exist, allowing them to correct their request or check the available tables.
        
        df = self.tables[table_name] # Get the DataFrame for the specified table name from the loaded tables. This allows us to access the data for that table in order to generate summary statistics using the pandas describe() method.
        return f"Summary for {table_name}:\n{df.describe().to_string()}" # Use the pandas describe() method to generate summary statistics for the specified table's DataFrame and return it as a formatted string. This provides insights into the distribution of values, counts, means, standard deviations, and other relevant statistics for the columns in that table, which can be useful for the AI agent 
                                                                         # when answering questions based on the data in that table. If the table contains non-numeric columns, the describe() method will provide summary statistics for those as well, such as counts, unique values, top values, and frequency for categorical data. This allows the AI agent to have a better understanding of the data when formulating responses to user questions based on the contents of the specified table.
    



    # get_data_context summarizes all loaded data for use in LLM context.
    # It iterates through all loaded tables, appending their names, columns, and a string representation of their data to a list. 
    # The resulting context is returned as a single formatted string. 
    # This method provides the AI agent with a comprehensive overview of all available data when answering questions, allowing it to reference specific tables and their contents in its responses. 
    def get_data_context(self) -> str:  
        """
        Get a summary of all data for LLM context.
        
         - Iterates through all loaded tables and appends their names, columns, and a string representation of their data to a list.         
        
        """
        context_parts = [] # Initialize an empty list to hold the different parts of the data context that will be built up as we iterate through the loaded tables. This allows us to construct a comprehensive overview of all available data for the AI agent when answering questions.
        
        for table_name, df in self.tables.items(): # Iterate through all loaded tables in the self.tables dictionary, where table_name is the name of the table and df is the corresponding pandas DataFrame containing the data for that table. This allows us to access each table's data and metadata to include in the context for the LLM.
            context_parts.append(f"\n=== {table_name} ===") # Append a header for the current table to the context_parts list, using the table name to clearly indicate which table's data is being summarized. This helps to organize the context and make it easier for the AI agent to reference specific tables when formulating responses to user questions based on the available data.
            context_parts.append(f"Columns: {list(df.columns)}") # Append a line to the context_parts list that lists the columns of the current table, providing information about the structure of the data in that table. This allows the AI agent to understand what data is available in each table and reference specific columns when answering questions based on the data.
            # For large tables, include summary stats and a sample instead of all rows to keep the LLM prompt manageable
            if len(df) > 50:
                context_parts.append(f"Total rows: {len(df)}")
                context_parts.append(f"Summary:\n{df.describe().to_string()}")
                context_parts.append(f"Sample data (first 20 rows):\n{df.head(20).to_string()}")
            else:
                context_parts.append(f"Data:\n{df.to_string()}")
        
        return "\n".join(context_parts) # Join all the parts of the context together into a single string with newline characters separating each part, and return it as the final data context for the LLM. 
                                        # This comprehensive context includes the names of all loaded tables, their columns, and a string representation of their data, providing the AI agent with a rich source of information to reference when answering questions based on the available data. The resulting context can be quite large if there are many tables or if the tables contain a lot of data, 
                                        # so it may be necessary to consider ways to summarize or limit the amount of data included in the context for larger datasets to ensure that it remains manageable for the LLM to process effectively.








# AIAgent class is responsible for using the Ollama LLM to answer questions about the e-commerce data.
# # It initializes with a DataQueryEngine to access the data and configures the Ollama LLM for generating responses. 
# The agent provides a method to answer questions by building a prompt that includes the data context and schema information, 
# and then invoking the LLM to generate a response based on that prompt. 
# If the LLM is not available or encounters an error, 
# it falls back to providing raw data or schema information as a response. 
# This allows the AI agent to still provide useful information even if the LLM is not functioning properly, ensuring that users can get insights from their data regardless of the status of the LLM. 
# The agent is designed to be interactive, allowing users to ask questions in natural language and receive answers based on their e-commerce data, 
#  making it a powerful tool for data analysis and insights generation without needing to write complex SQL queries directly.                                         
class AIAgent:
    """
    AI Agent that uses Ollama to answer questions about e-commerce data.
        - Initializes with a DataQueryEngine to access the data.
        - Configures the Ollama LLM for generating responses.
        - Provides a method to answer questions by building a prompt that includes the data context and schema information, and then invoking the LLM to generate a response based on that prompt. 
        - If the LLM is not available or encounters an error, it falls back to providing raw data or schema information as a response. This allows the AI agent to still provide useful information even if the LLM is not functioning properly, ensuring that users can get insights from their data regardless of the status of the LLM. 
        - The agent is designed to be interactive, allowing users to ask questions in natural language and receive answers based on their e-commerce data, making it a powerful tool for data analysis and insights generation without needing to write complex SQL queries directly.
    
    """
    
    def __init__(self, query_engine: DataQueryEngine):
        self.query_engine = query_engine # Store the provided DataQueryEngine instance in the agent for accessing the data when answering questions. This allows the agent to query the data and include relevant information in the prompts sent to the LLM, enabling it to generate informed responses based on the available data.
        self.llm = None # Initialize the LLM attribute to None, which will later be set to an instance of the OllamaLLM if it is successfully configured. This allows us to check if the LLM is available before attempting to invoke it for generating responses, and to provide fallback responses if the LLM is not configured properly or encounters errors.
        self._init_llm() # Call the method to initialize the Ollama LLM when the agent is created. This will attempt to set up the LLM and print the status of the configuration, allowing us to know if the LLM is ready to use for answering questions or if we need to rely on fallback responses based on the data context and schema information.
    
    def _init_llm(self):
        """
        Initialize the Ollama LLM.

            - Attempts to create an instance of the OllamaLLM with the specified model and base URL.
            - If successful, it prints a confirmation message indicating that Ollama is configured.
            - If there is an error during initialization (e.g., Ollama server not running, incorrect model name, connection issues), 
            it catches the exception and prints an error message with details about the failure. 
            It also provides a reminder to ensure that the Ollama server is running (e.g., "ollama serve") to help users troubleshoot common issues with LLM configuration. 
            If the initialization fails the LLM attribute will remain None, allowing the agent to fall back to providing raw data or schema information when answering questions instead of generating responses from the LLM. 
            This ensures that the agent can still provide useful insights based on the data even if the LLM is not available.         


        """
        try:
            self.llm = OllamaLLM(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=0.1
            ) # Attempt to create an instance of the OllamaLLM with the specified model, base URL, and temperature settings. If this is successful, it means that the LLM is properly configured and ready to use for generating responses to user questions based on the e-commerce data.      
            print(f"✓ Configured Ollama ({OLLAMA_MODEL})")
        except Exception as e: 
            print(f"✗ Failed to configure Ollama: {e}")
            print("  Make sure Ollama is running: ollama serve")
            self.llm = None




    # answer_question is the main method that takes a user's question as input and generates an answer based on the e-commerce data.
    # It first retrieves the data context and schema information from the DataQueryEngine, then builds a prompt that includes this information along with the user's question. 
    # The prompt is designed to instruct the LLM to provide a clear and concise answer based on the data, including specific numbers and insights. 
    # If the LLM is available, it invokes the LLM with the prompt and returns the generated response. 
    # If there is an error during LLM invocation, it catches the exception and returns an error message along with a fallback response that includes the table schemas.             
    def answer_question(self, question: str) -> str:
        """
        Answer a question about the e-commerce data.

        - Retrieves the data context and schema information from the DataQueryEngine.
        - Builds a prompt that includes the data context, schema information, and the user's question.
        - The prompt instructs the LLM to provide a clear and concise answer based on the data, including specific numbers and insights. 
          If the data is insufficient, it instructs the LLM to say so clearly.

        - If the LLM is available, it invokes the LLM with the prompt and returns the generated response. 
        - If there is an error during LLM invocation, it catches the exception and returns an error message along with a fallback response 
          that includes the table schemas. This allows the agent to still provide useful information based on the data even if the LLM encounters issues, 
          ensuring that users can get insights from their e-commerce data regardless of the status of the LLM. 

        - If the LLM is not available at all, it returns a message indicating that the LLM is not available and provides the raw data context as a fallback response, 
          allowing users to still see the data that the agent has access to even 
          if it cannot generate a response using the LLM. This ensures that the agent can still be useful for data exploration and insights generation 
          even in the absence of a functioning LLM, providing users with direct access to the data context and schema information that the agent uses to answer questions.          
        
        
        """
        
        # Get data context
        data_context = self.query_engine.get_data_context() # Retrieve the data context from the DataQueryEngine, which provides a summary of all loaded data including table names, columns, and a string representation of the data. This context will be included in the prompt sent to the LLM to give it the necessary information about the available data when generating a response to the user's question.
        schema_info = self.query_engine.get_table_schemas() # Retrieve the schema information for all loaded tables from the DataQueryEngine, which provides details about the columns, data types, and row counts for each table. This schema information will be included in the prompt sent to the LLM to give it a clear understanding of the structure of the data it can reference when answering the user's question. By including both the data context and schema information in the prompt, we enable the LLM to generate more informed and accurate responses based on the available e-commerce data, allowing it to provide specific insights and numbers as needed to answer the user's question effectively.
        
        # Build prompt
        prompt = f"""You are an AI data analyst for an e-commerce platform.
Answer the user's question based on the data provided below.

AVAILABLE TABLES: 
{schema_info}

CURRENT DATA:
{data_context}

USER QUESTION: {question}

Provide a clear, concise answer based on the data above. Include specific numbers and insights.
If the data is insufficient, say so clearly.

ANSWER:"""
        

# If the LLM is available, we will attempt to invoke it with the constructed prompt to generate a response to the user's question based on the e-commerce data. 
# This allows us to leverage the capabilities of the LLM to provide a more natural and informative answer that references specific insights from the data. 
# If there is an error during LLM invocation, we will catch the exception and return an error message along with a fallback response that includes the table schemas, ensuring that users can still get useful information about the available data even if the LLM encounters issues. 
# If the LLM is not available at all, we will return a message indicating that the LLM is not available and provide the raw data context as a fallback response, allowing users to still see the data that the agent has access to even if it cannot generate a response using the LLM. 
# This ensures that the agent can still be useful for data exploration and insights generation even in the absence of a functioning LLM, providing users with direct access to the data context and schema information that the agent uses to answer questions.

        if self.llm:  
            try:
                print("  🤔 Thinking...")
                response = self.llm.invoke(prompt) # Invoke the LLM with the constructed prompt to generate a response based on the e-commerce data and the user's question. This allows us to leverage the LLM's capabilities to provide a more natural and informative answer that references specific insights from the data. If the invocation is successful, we will return the generated response as the answer to the user's question.
                return response.strip() # Return the generated response from the LLM, stripping any leading or trailing whitespace for cleaner output. This will be the answer to the user's question based on the e-commerce data and the information provided in the prompt. If the LLM is able to generate a response successfully, it should provide insights and specific numbers based on the data context and schema information included in the prompt.
            except Exception as e:
                return f"LLM Error: {str(e)}\n\nFalling back to data display:\n{self.query_engine.get_table_schemas()}"
        else:
            return f"LLM not available. Here's the raw data:\n{data_context}"



# run_interactive_session is a function that runs an interactive question-answering session with the user.
# It initializes the DataQueryEngine and AIAgent, then enters a loop where it prompts the user to ask questions about their e-commerce data in natural language. 
# The user can also enter direct SQL queries by prefixing their input with "sql ". The function handles user input, processes it accordingly (either as a direct SQL query or as a natural language question), and prints the results. 
# The session continues until the user types "quit", "exit", or "q".        
def run_interactive_session():
    """
    Run an interactive question-answering session.

        - Initializes the DataQueryEngine and AIAgent.
        - Prompts the user to ask questions about their e-commerce data in natural language.
        - The user can also enter direct SQL queries by prefixing their input with "sql ".
        - Handles user input, processes it accordingly (either as a direct SQL query or as a natural language question), and prints the results.
        - The session continues until the user types "quit", "exit", or "q".    
    
    
    """
    print("=" * 60)
    print("🤖 E-commerce AI Data Agent")
    print("=" * 60)
    print("Ask questions about your e-commerce data in natural language.")
    print("Commands: 'quit' to exit, 'sql <query>' for direct SQL")
    print("=" * 60)
    
    # Initialize
    query_engine = DataQueryEngine() # Create an instance of the DataQueryEngine, which will be responsible for loading the e-commerce data and providing methods to query that data. This engine will allow us to access the data in a structured way and provide the necessary context and schema information to the AI agent when answering questions.
    agent = AIAgent(query_engine) # Create an instance of the AIAgent, passing in the DataQueryEngine instance. The agent will use the query engine to access the data and generate responses to user questions based on that data. This allows us to have a modular design where the agent is responsible for interacting with the LLM and generating answers, while the query engine is responsible for managing the data and providing access to it when needed.
    
    print("\n📝 Example questions:")
    print("  • What is the total revenue?")
    print("  • Show me the conversion rates")
    print("  • How many active users do we have?")
    print("  • sql SELECT * FROM revenue_per_hour")
    print()
    
    while True: # Start an infinite loop to continuously prompt the user for input until they choose to exit. This allows for an interactive session where users can ask multiple questions about their e-commerce data without needing to restart the program.
        try:
            question = input("\n🤖 Ask a question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            print("\n" + "-" * 50)
            
            # Handle direct SQL queries
            if question.lower().startswith("sql "): # If the user's input starts with "sql ", we will treat it as a direct SQL query that they want to execute against the data. This allows users who are familiar with SQL to run specific queries directly without needing to phrase their question in natural language, giving them more control over the data they want to retrieve or analyze.
                sql_query = question[4:].strip() # Extract the SQL query from the user's input by removing the "sql " prefix and stripping any leading or trailing whitespace. This gives us the actual SQL query that the user wants to execute against the data.
                result = query_engine.execute_query(sql_query) # Use the DataQueryEngine's execute_query method to run the extracted SQL query against the loaded data. This method will attempt to parse and execute the SQL query, returning the results as a string. If the query is valid and can be executed successfully, it will return the results of that query based on the data available in the DataQueryEngine. If there are any errors during query execution (e.g., syntax errors, referencing non-existent tables or columns), it will return an error message indicating what went wrong with the query.
                print(f"\n📊 SQL Result:\n{result}") 
            elif question.lower() in ['describe', 'show tables', 'schema']: # If the user's input is a command to show the schema or available tables (e.g., "describe", "show tables", "schema"), we will respond by providing the table schemas from the DataQueryEngine. This allows users to quickly see what tables are available and their structure, which can help them formulate their questions or SQL queries more effectively based on the data they have access to.
                print(f"\n📊 Available Tables:\n{query_engine.get_table_schemas()}") 
            else:
                # Use AI to answer
                answer = agent.answer_question(question) # If the user's input is not a direct SQL query or a command to show the schema, we will treat it as a natural language question that they want the AI agent to answer based on the e-commerce data. We will pass the question to the agent's answer_question method, which will build a prompt that includes the data context and schema information, and then invoke the LLM to generate a response based on that prompt. The generated response will be returned as the answer to the user's question, providing insights and specific numbers based on the data available in the DataQueryEngine. 
                                                         # If there are any issues with the LLM (e.g., it is not available, or there is an error during invocation), the agent will provide fallback responses based on the data context and schema information to ensure that users can still get useful insights from their data even if the LLM encounters problems.
                print(f"\n📊 Answer:\n{answer}") 
                
        except KeyboardInterrupt: # If the user sends a keyboard interrupt signal (e.g., by pressing Ctrl+C), we will catch that exception and print a goodbye message before breaking out of the loop to end the interactive session gracefully. This allows users to exit the session at any time using a common keyboard shortcut, providing a convenient way to end their interaction with the AI agent when they are finished asking questions about their e-commerce data.
            print("\n\n👋 Goodbye!")
            break
        except Exception as e: # If there are any other exceptions that occur during the processing of user input (e.g., issues with the DataQueryEngine, unexpected errors in the agent's methods), we will catch those exceptions and print an error message with details about what went wrong. This helps to provide feedback to the user if something goes wrong during their interaction with the AI agent, allowing them to understand that an error occurred and potentially take corrective action or try a different question. By catching general exceptions, we can ensure that the interactive session continues even if there are issues, rather than crashing the program, providing a more robust user experience.
            print(f"❌ Error: {e}")


def run_agent(question: str) -> dict:
    """
    Run the AI agent with a question and return structured results.

    Args:
        question: Natural language question about the e-commerce data.

    Returns:
        dict with keys:
            - sql: The SQL-like query derived from the question (if applicable).
            - data: A pandas DataFrame with the result data.
            - summary: LLM-generated explanation of the results.
    """
    query_engine = DataQueryEngine()
    agent = AIAgent(query_engine)

    summary = ""
    sql = ""
    data = pd.DataFrame()

    # Check if it's a direct SQL query
    if question.lower().startswith("sql "):
        sql = question[4:].strip()
        result_str = query_engine.execute_query(sql)
        # Try to get a DataFrame from the matched table
        from_match = re.search(r'from\s+(\w+)', sql, re.IGNORECASE)
        if from_match:
            table_name = from_match.group(1)
            for t in query_engine.tables:
                if t.lower() == table_name.lower() or table_name.lower() in t.lower():
                    data = query_engine.tables[t].copy()
                    limit_match = re.search(r'limit\s+(\d+)', sql, re.IGNORECASE)
                    if limit_match:
                        data = data.head(int(limit_match.group(1)))
                    else:
                        data = data.head(100)
                    break
        summary = result_str
    elif question.lower() in ['describe', 'show tables', 'schema']:
        summary = query_engine.get_table_schemas()
    else:
        # Use AI to answer
        summary = agent.answer_question(question)
        # Try to find the most relevant table for the question
        q_lower = question.lower()
        for table_name, df in query_engine.tables.items():
            if any(keyword in q_lower for keyword in table_name.lower().split('_')):
                data = df.copy()
                break
        # If no table matched, provide all gold data
        if data.empty and query_engine.tables:
            first_table = list(query_engine.tables.keys())[0]
            data = query_engine.tables[first_table].copy()

    return {
        "sql": sql,
        "data": data,
        "summary": summary,
    }


def main(): # Define the main function as the entry point of the program.
    """
    Main entry point.

        - Calls the run_interactive_session function to start the interactive question-answering session with the user. 
          This allows us to keep the main function clean and focused on starting the program, while the run_interactive_session function handles all the details of the interactive session itself, 
          including initializing the DataQueryEngine and AIAgent, prompting the user for input, and processing that input accordingly. By structuring the program this way, we can maintain a clear separation of concerns and keep our code organized and easy to understand.          
    
    """
    run_interactive_session() # Call the function to run the interactive session, which will start the program and allow users to ask questions about their e-commerce data in natural language, as well as execute direct SQL queries if they choose to do so. This is the main functionality of the program, providing an interactive interface for users to explore their data and get insights based on their questions.




if __name__ == "__main__":
    main()
