import hashlib

class Document:
    def __init__(self, title=None, text=None):
        if title == None or text == None:
            self.__title = None
            self.__text = None
        else:
            self.__title = title
            self.__text = text
        self.__hash = None
        self.__value = None
    def __hash__(self):
        if self.__hash:
            return self.__hash
        else:
            h = hashlib.md5()
            h.update(bytes(self.__title))
            h.update(bytes(self.__text))
            self.__hash = h.hexdigest()
            return self.__hash
    def compute_val(self, classifier):
        if self.__value:
            return self.__value
        else:
            self.__value = classifier.classify(self.__title, self.__text)
            print(self.__value)
            return self.__value