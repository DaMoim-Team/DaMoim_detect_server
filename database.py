import mysql.connector
from config_DB import Config_DB

def update_database(detect_values):
    connection = None
    cursor = None
    try:
        print("Connecting to the database...")
        connection = mysql.connector.connect(
            host=Config_DB.HOST,
            port=Config_DB.PORT,
            user=Config_DB.USER,
            password=Config_DB.PASSWORD,
            database=Config_DB.DATABASE
        )
        print("Connected to the database.")

        cursor = connection.cursor()

        # detect_values 딕셔너리의 각 항목에 대해 SQL 쿼리문을 생성하고 실행
        for cctv, value in detect_values.items():
            sql_query = f"UPDATE locations SET count_cleanup = count_cleanup + {value}, count_catch = count_catch + {value} WHERE cctv_id = '{cctv}'"
            cursor.execute(sql_query)

        # 변경사항 커밋
        connection.commit()

    except Exception as e:
        print(f"Error while connecting to MySQL: {str(e)}")

    finally:
        if (connection and connection.is_connected()):
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

def update_ranktable(oneday_dict):
    connection = None
    cursor = None
    try:
        print("Connecting to the database...")
        connection = mysql.connector.connect(
            host=Config_DB.HOST,
            port=Config_DB.PORT,
            user=Config_DB.USER,
            password=Config_DB.PASSWORD,
            database=Config_DB.DATABASE
        )
        print("Connected to the database.")
        
        cursor = connection.cursor()
        
        for time_range, values in oneday_dict.items():
            most, top_count = values
            sql_query = f"UPDATE timeranking SET most = '{most}', top_count = {top_count} WHERE time_range = '{time_range}'"
            cursor.execute(sql_query)
        
        # Commit the changes
        connection.commit()

    except Exception as e:
        print(f"Error while connecting to MySQL: {str(e)}")
        
    finally:
        if (connection and connection.is_connected()):
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
