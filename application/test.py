
import pyodbc, os
import dotenv
dotenv.load_dotenv()
# set up the connection string
server = os.getenv('AZURE_SQL_SERVER')
database = os.getenv('AZURE_SQL_DATABASE')
username = os.getenv('AZURE_SQL_USERNAME')
password = os.getenv('AZURE_SQL_PASSWORD')
cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)

# create a cursor and execute a query
cursor = cnxn.cursor()
cursor.execute("SELECT * FROM PlayListTrack")

# fetch the results
row = cursor.fetchone()
while row:
    print(row)
    row = cursor.fetchone()