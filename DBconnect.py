import MySQLdb

def connection():
    conn = MySQLdb.connect(host="",
                           user = "",
                           passwd = "",
                           db = "")
    c = conn.cursor()
    return c, conn
