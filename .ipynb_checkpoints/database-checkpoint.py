import mysql.connector

db_config = {
    "user": "root",
    "password": "a7879310",
    "host": "localhost",
    "database": "LINE",
}

def get_allowed_users_count(line_id):
    cnx = mysql.connector.connect(**db_config)
    cursor = cnx.cursor()
    query = "SELECT COUNT(*) FROM allowed_users WHERE line_id = %s"
    cursor.execute(query, (line_id,))
    count = cursor.fetchone()[0]
    cursor.close()
    cnx.close()
    return count
