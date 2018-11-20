import logging

class LOG:
    def __init__(self,file,name):
        logging.basicConfig(level=logging.INFO,
                            filename=file,
                            filemode='a',
                            format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                            )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        self._log = logging.getLogger(name)
        self._log.addHandler(console)
        self._log.info("\n\n********NEW RECORD********\n")


    def getlogger(self):
        return self._log

