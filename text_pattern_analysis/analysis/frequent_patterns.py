"""
Frequent pattern mining for text analysis
"""

from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth

def create_spark_session(app_name="TextPatternAnalysis"):
    """
    Create a Spark session for pattern mining
    
    Args:
        app_name (str): Name of the Spark application
        
    Returns:
        pyspark.sql.SparkSession: Configured Spark session
    """
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

def prepare_data_for_pattern_mining(tokens_list):
    """
    Prepare tokenized text data for pattern mining in Spark
    
    Args:
        tokens_list (list): List of lists, where each inner list contains tokens from one document
        
    Returns:
        pyspark.sql.DataFrame: DataFrame prepared for pattern mining
    """
    spark = create_spark_session()
    
    # Convert to Spark DataFrame
    df = spark.createDataFrame([(tokens,) for tokens in tokens_list], ["items"])
    
    return df

def mine_frequent_patterns(df, min_support=0.01, min_confidence=0.5):
    """
    Mine frequent patterns and association rules from text data
    
    Args:
        df (pyspark.sql.DataFrame): DataFrame containing tokenized items
        min_support (float): Minimum support threshold
        min_confidence (float): Minimum confidence threshold
        
    Returns:
        tuple: (frequent_itemsets_df, association_rules_df)
    """
    # Create FP-Growth model
    fpgrowth = FPGrowth(itemsCol="items", 
                       minSupport=min_support, 
                       minConfidence=min_confidence)
    
    # Train model
    model = fpgrowth.fit(df)
    
    # Get frequent itemsets and association rules
    frequent_itemsets = model.freqItemsets
    association_rules = model.associationRules
    
    return frequent_itemsets, association_rules

def convert_spark_results_to_pandas(frequent_itemsets, association_rules):
    """
    Convert Spark DataFrames to Pandas DataFrames for easier analysis
    
    Args:
        frequent_itemsets (pyspark.sql.DataFrame): Frequent itemsets from FP-Growth
        association_rules (pyspark.sql.DataFrame): Association rules from FP-Growth
        
    Returns:
        tuple: (itemsets_pandas_df, rules_pandas_df)
    """
    # Convert to Pandas
    itemsets_pandas = frequent_itemsets.toPandas()
    rules_pandas = association_rules.toPandas()
    
    return itemsets_pandas, rules_pandas 