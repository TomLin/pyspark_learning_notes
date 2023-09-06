df = spark.read.format("csv")\
  .option("header", "true")\
  .option("inferSchema", "true")\
  .load("/databricks-datasets/definitive-guide/data/retail-data/by-day/2010-12-01.csv")
df.printSchema()
df.createOrReplaceTempView("dfTable")


# COMMAND ----------

from pyspark.sql.functions import lit
df.select(lit(5), lit("five"), lit(5.0))
# noted:
# we always need to convert the data from source data-type to spark data-type

# COMMAND ----------

from pyspark.sql.functions import col
df.where(col("InvoiceNo") != 536365)\
  .select("InvoiceNo", "Description")\
  .show(5, False)

# Arguments:
# show(n=20, truncate=True)


# COMMAND ----------

# create two filters for use later
# when we join multiple conditions, we can chain `and` condition in multiple statement, 
# but we must need to place `or` condition in the same statement.
from pyspark.sql.functions import instr
priceFilter = col("UnitPrice") > 600
descripFilter = instr(df.Description, "POSTAGE") >= 1
df.where(df.StockCode.isin("DOT")).where(priceFilter | descripFilter).show()

# instr:
# Locate the position of the first occurrence of substr column in the given string. Returns null if either of the arguments are null.
# The position is not zero based, but 1 based index. Returns 0 if substr could not be found in str.

# COMMAND ----------

# we can also use these conditional statements to create new boolean column
from pyspark.sql.functions import instr
DOTCodeFilter = col("StockCode") == "DOT"
priceFilter = col("UnitPrice") > 600
descripFilter = instr(col("Description"), "POSTAGE") >= 1
df.withColumn("isExpensive", DOTCodeFilter & (priceFilter | descripFilter))\
  .where("isExpensive")\
  .select("unitPrice", "isExpensive").show(5)

# Noted:
# when we join two `and` statements, we use `&` operator

# COMMAND ----------

from pyspark.sql.functions import expr
df.withColumn("isExpensive", expr("NOT UnitPrice <= 250"))\
  .where("isExpensive")\
  .select("Description", "UnitPrice").show(5)


# COMMAND ----------

from pyspark.sql.functions import expr, pow
fabricatedQuantity = pow(col("Quantity") * col("UnitPrice"), 2) + 5
df.select(expr("CustomerId"), fabricatedQuantity.alias("realQuantity")).show(2)

# pow:
# Returns the value of the first argument raised to the power of the second argument.

# COMMAND ----------

df.selectExpr(
  "CustomerId",
  "(POWER((Quantity * UnitPrice), 2.0) + 5) as realQuantity").show(2)


# COMMAND ----------

from pyspark.sql.functions import lit, round, bround

df.select(round(lit("2.5")), bround(lit("2.5"))).show(2)

# `bround` is the reverse of `round`, it will round down to specified decimal or integer

# COMMAND ----------

from pyspark.sql.functions import corr
df.stat.corr("Quantity", "UnitPrice")
df.select(corr("Quantity", "UnitPrice")).show()

# we can compute the correlation of two numerical columns either through
# a function or through `stat` method in DataFrame


# COMMAND ----------

df.describe().show()


# COMMAND ----------

from pyspark.sql.functions import count, mean, stddev_pop, min, max


# COMMAND ----------

colName = "UnitPrice"
quantileProbs = [0.5]
relError = 0.05
df.stat.approxQuantile("UnitPrice", quantileProbs, relError) # 2.51

# approxQuantile(col, probabilities, relativeError):
# col – str, list. Can be a single column name, or a list of names for multiple columns.
# probabilities – a list of quantile probabilities Each number must belong to [0, 1].
# relativeError – The relative target precision to achieve (>= 0). 
# If set to zero, the exact quantiles are computed, which could be very expensive. 
# Note that values greater than 1 are accepted but give the same result as 1.


# COMMAND ----------

df.stat.crosstab("StockCode", "Quantity").show()


# COMMAND ----------

df.stat.freqItems(["StockCode", "Quantity"]).show()


# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id
df.select(monotonically_increasing_id()).show(2)


# COMMAND ----------

from pyspark.sql.functions import initcap
df.select(initcap(col("Description"))).show()

# initcap: capitalize every word in a given string

# COMMAND ----------

from pyspark.sql.functions import lower, upper
df.select(col("Description"),
    lower(col("Description")),
    upper(lower(col("Description")))).show(2)


# COMMAND ----------

from pyspark.sql.functions import lit, ltrim, rtrim, rpad, lpad, trim
df.select(
    ltrim(lit("    HELLO    ")).alias("ltrim"),
    rtrim(lit("    HELLO    ")).alias("rtrim"),
    trim(lit("    HELLO    ")).alias("trim"),
    lpad(lit("HELLO"), 3, " ").alias("lp"),
    rpad(lit("HELLO"), 10, " ").alias("rp")).show(2)


# COMMAND ----------

# Spark uses Java regular expressions
# The most two important functions in regular expression: regexp_extract, regexp_replace

from pyspark.sql.functions import regexp_replace
regex_string = "BLACK|WHITE|RED|GREEN|BLUE"
df.select(
  regexp_replace(col("Description"), regex_string, "COLOR").alias("color_clean"),
  col("Description")).show(2)

#regexp_replace(str, pattern, replacement): Replace all substrings.

# COMMAND ----------

from pyspark.sql.functions import translate
df.select(translate(col("Description"), "LEET", "1337"),col("Description"))\
  .show(2)

# translate(srcCol, matching, replace): similar to regexp_replace, but it is done in character level,
# the pors is that it can replace a set of characters at the same time.

# COMMAND ----------

from pyspark.sql.functions import regexp_extract
extract_str = "(BLACK|WHITE|RED|GREEN|BLUE)"
df.select(
     regexp_extract(col("Description"), extract_str, 1).alias("color_clean"),
     col("Description")).show(2)

# regexp_extract(str, pattern, idx): If the regex did not match, or the specified group did not match, an empty string is returned.

# COMMAND ----------

from pyspark.sql.functions import instr
containsBlack = instr(col("Description"), "BLACK") >= 1
containsWhite = instr(col("Description"), "WHITE") >= 1
df.withColumn("hasSimpleColor", containsBlack | containsWhite)\
  .where("hasSimpleColor")\
  .select("Description").show(3, False)


# COMMAND ----------

from pyspark.sql.functions import expr, locate
simpleColors = ["black", "white", "red", "green", "blue"]
def color_locator(column, color_string):
  return locate(color_string.upper(), column)\
          .cast("boolean")\
          .alias("is_" + color_string)
selectedColumns = [color_locator(df.Description, c) for c in simpleColors]
selectedColumns.append(expr("*")) # has to a be Column type

df.select(*selectedColumns).where(expr("is_white OR is_red"))\
  .select("Description").show(3, False)

# locate(substr, str, pos=1): Locate the position of the first occurrence of substr in a string column, after position pos.
# The position is not zero based, but 1 based index. Returns 0 if substr could not be found in str.


# COMMAND ----------

from pyspark.sql.functions import current_date, current_timestamp
dateDF = spark.range(10)\
  .withColumn("today", current_date())\
  .withColumn("now", current_timestamp())
dateDF.createOrReplaceTempView("dateTable")


# COMMAND ----------

from pyspark.sql.functions import date_add, date_sub
dateDF.select(date_sub(col("today"), 5), date_add(col("today"), 5)).show(1)


# COMMAND ----------

from pyspark.sql.functions import datediff, months_between, to_date
dateDF.withColumn("week_ago", date_sub(col("today"), 7))\
  .select(datediff(col("week_ago"), col("today"))).show(1)

dateDF.select(
    to_date(lit("2016-01-01")).alias("start"),
    to_date(lit("2017-05-22")).alias("end"))\
  .select(months_between(col("start"), col("end"))).show(1)


# COMMAND ----------

from pyspark.sql.functions import to_date, lit
spark.range(5).withColumn("date", lit("2017-01-01"))\
  .select(to_date(col("date"))).show(1)

# The to_date function allows you to convert a string to a date, 
# optionally with a specified format. We specify our format in the Java SimpleDateFormat 
# which will be important to reference if you use this function.

# Spark will not throw an error if it cannot parse the date; rather, it will just return null.


# COMMAND ----------

from pyspark.sql.functions import to_date
dateFormat = "yyyy-dd-MM"
cleanDateDF = spark.range(1).select(
    to_date(lit("2017-12-11"), dateFormat).alias("date"),
    to_date(lit("2017-20-12"), dateFormat).alias("date2"))
cleanDateDF.createOrReplaceTempView("dateTable2")


# COMMAND ----------

from pyspark.sql.functions import to_timestamp
cleanDateDF.select(to_timestamp(col("date"), dateFormat)).show()

# to_timestamp is different from to_date, for it requires dateFormat explicitly.

# COMMAND ----------

from pyspark.sql.functions import coalesce
df.select(coalesce(col("Description"), col("CustomerId"))).show()


# COMMAND ----------

df.na.drop("all", subset=["StockCode", "InvoiceNo"])

# df.dropna() is an alias to df.na.drop()
# df.dropna() is a newly-added method

# dropna(how='any', thresh=None, subset=None):
# how – ‘any’ or ‘all’. If ‘any’, drop a row if it contains any nulls. If ‘all’, drop a row only if all its values are null.
# thresh – int, default None If specified, drop rows that have less than thresh non-null values. This overwrites the how parameter.
# subset – optional list of column names to consider.

# COMMAND ----------

df.na.fill("all", subset=["StockCode", "InvoiceNo"])
# fill all null value with the string `all`

# df.fillna() is an alias to df.na.fill()
# fillna(value, subset=None)

# COMMAND ----------

fill_cols_vals = {"StockCode": 5, "Description" : "No Value"}
df.na.fill(fill_cols_vals)


# COMMAND ----------

df.na.replace([""], ["UNKNOWN"], "Description")

# df.replace() is an alias to df.na.replace()
# replace(to_replace, value=<no value>, subset=None) 
# the value can also be None

# COMMAND ----------

from pyspark.sql.functions import struct
complexDF = df.select(struct("Description", "InvoiceNo").alias("complex"))
complexDF.createOrReplaceTempView("complexDF")

complexDF.select("complex.Description").show(2, False)
complexDF.select(col("complex").getField("Description")).show(2, False)
complexDF.select("complex.*") # select all fields at once


# COMMAND ----------

from pyspark.sql.functions import split
df.select(split(col("Description"), " ")).show(2) # convert into an array


# COMMAND ----------

df.select(split(col("Description"), " ").alias("array_col"))\
  .selectExpr("array_col[0]").show(2)


# COMMAND ----------

from pyspark.sql.functions import size # size is used to compute array length
df.select(size(split(col("Description"), " "))).show(2) # shows 5 and 3


# COMMAND ----------

from pyspark.sql.functions import array_contains
df.select(array_contains(split(col("Description"), " "), "WHITE")).show(2)


# COMMAND ----------

from pyspark.sql.functions import split, explode

df.withColumn("splitted", split(col("Description"), " "))\
  .withColumn("exploded", explode(col("splitted")))\
  .select("Description", "InvoiceNo", "exploded").show(2)

# The explode function takes a column that consists of arrays 
# and creates one row (with the rest of the values duplicated) per value in the array.

# COMMAND ----------

from pyspark.sql.functions import create_map
df.select(create_map(col("Description"), col("InvoiceNo")).alias("complex_map"))\
  .show(2)

# the key will be the value from column `Description`
# the value will be the value from column `InvoiceNo`

# COMMAND ----------

df.select(create_map(col("Description"), col("InvoiceNo")).alias("complex_map"))\
  .selectExpr("complex_map['WHITE METAL LANTERN']").show(2)

# the missing key will return Null

# COMMAND ----------

df.select(create_map(col("Description"), col("InvoiceNo")).alias("complex_map"))\
  .selectExpr("explode(complex_map)").show(2)

# You can also explode map types, which will turn them into columns

# COMMAND ----------

jsonDF = spark.range(1).selectExpr("""
  '{"myJSONKey" : {"myJSONValue" : [1, 2, 3]}}' as jsonString""")

# Output:
# +-------------------------------------------+
# |jsonString                                 |
# +-------------------------------------------+
# |{"myJSONKey" : {"myJSONValue" : [1, 2, 3]}}|
# +-------------------------------------------+


# COMMAND ----------

from pyspark.sql.functions import get_json_object, json_tuple

jsonDF.select(
    get_json_object(col("jsonString"), "$.myJSONKey.myJSONValue[1]").alias("column"),
    json_tuple(col("jsonString"), "myJSONKey")).show(2)

# json_tuple is used if the object has only one level of nesting.

# Output:
# +------+-----------------------+
# |column|c0                     |
# +------+-----------------------+
# |2     |{"myJSONValue":[1,2,3]}|
# +------+-----------------------+


# COMMAND ----------

from pyspark.sql.functions import to_json
df.selectExpr("(InvoiceNo, Description) as myStruct")\
  .select(to_json(col("myStruct")), col("myStruct"))

# Output:
# +-------------------------------------------------------------------------+--------------------------------------------+
# |structstojson(myStruct)                                                  |myStruct                                    |
# +-------------------------------------------------------------------------+--------------------------------------------+
# |{"InvoiceNo":"536365","Description":"WHITE HANGING HEART T-LIGHT HOLDER"}|[536365, WHITE HANGING HEART T-LIGHT HOLDER]|
# |{"InvoiceNo":"536365","Description":"WHITE METAL LANTERN"}               |[536365, WHITE METAL LANTERN]               |
# +-------------------------------------------------------------------------+--------------------------------------------+


# COMMAND ----------

from pyspark.sql.functions import from_json
from pyspark.sql.types import *
parseSchema = StructType((
  StructField("InvoiceNo",StringType(),True),
  StructField("Description",StringType(),True)))
df.selectExpr("(InvoiceNo, Description) as myStruct")\
  .select(to_json(col("myStruct")).alias("newJSON"))\
  .select(from_json(col("newJSON"), parseSchema), col("newJSON")).show(2)

# from_json usually requires you to specify the schema
# `from_json(col("newJSON"), parseSchema)` will parse the jason data to a struct

# Output:
# +---------------------------------------------+--------------------------------------------------------------------------+
# |jsontostructs(newJSON)                       |newJSON                                                                   |
# +---------------------------------------------+--------------------------------------------------------------------------+
# |[536365, WHITE HANGING HEART T-LIGHT HOLDER] |{"InvoiceNo":"536365","Description":"WHITE HANGING HEART T-LIGHT HOLDER"} |
# |[536365, WHITE METAL LANTERN]                |{"InvoiceNo":"536365","Description":"WHITE METAL LANTERN"}                |
# |[536365, CREAM CUPID HEARTS COAT HANGER]     |{"InvoiceNo":"536365","Description":"CREAM CUPID HEARTS COAT HANGER"}     |
# |[536365, KNITTED UNION FLAG HOT WATER BOTTLE]|{"InvoiceNo":"536365","Description":"KNITTED UNION FLAG HOT WATER BOTTLE"}|
# |[536365, RED WOOLLY HOTTIE WHITE HEART.]     |{"InvoiceNo":"536365","Description":"RED WOOLLY HOTTIE WHITE HEART."}     |
# +---------------------------------------------+--------------------------------------------------------------------------+



# COMMAND ----------

udfExampleDF = spark.range(5).toDF("num")
def power3(double_value):
  return double_value ** 3
power3(2.0)


# COMMAND ----------

from pyspark.sql.functions import udf
power3udf = udf(power3)
# to register the function as DataFrame function

# COMMAND ----------

from pyspark.sql.functions import col
udfExampleDF.select(power3udf(col("num"))).show(2)


# COMMAND ----------

udfExampleDF.selectExpr("power3(num)").show(2)
# registered in Scala


# COMMAND ----------

from pyspark.sql.types import IntegerType, DoubleType
spark.udf.register("power3py", power3, DoubleType())
# to register the function as Spark SQL function

# noted, when we specify the return type of power3 to be DoubleType(),
# then if the function power3 don't return float (in python data-type),
# instead it returns integer (in python data-type),
# then when we use the function in spark sql, it will return null values.
# It is a good practice for us to spot any bugs.
# because we need to ensure that python's data-type is aligned to spark's data-type

# COMMAND ----------

udfExampleDF.selectExpr("power3py(num)").show(2)
# registered via Python


# COMMAND ----------

# The data-type user-defined function can return, and this is how we should define.
# It is always best practice to explicitly specify the return data type.
# Ensure that it's pythonic data type going to be aligned to Spark data type.
# String, Boolean, Date, Timestamp, Double, Float, Integer, Long, Short, Array, Map, Struct

