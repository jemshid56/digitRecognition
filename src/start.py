from datetime import datetime
f = open("report/starttime", "w")
f.write('Start Printed string Recorded at: %s\n' %datetime.now())
f.close()

