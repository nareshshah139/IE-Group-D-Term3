#pyspark --packages com.databricks:spark-xml_2.11:0.3.3
import xml.etree.ElementTree as ET
from pyspark.sql import Row

def xmltordd(xmlstring):
  dict = {"entailment":None, "text":None,"task":None,"pair" : None,"hypothesis":None}
  tree = ET.parse(xmlstring)
  root = tree.getroot()
  for child in root:
    dict["entailment"] = child.attrib.get("entailment")
    dict["text"]= child.find("t").text
    dict["task"]= child.attrib.get("task")
    dict["pair"]= child.attrib.get("pair")
    dict["hypothesis"] = child.find("h").text
   
  rdd1 = sc.parallelize(dict)
  return rdd1
rdd1 = sc.textFile().map(xmltordd)
