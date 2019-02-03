import hashlib

class Document:
    def __init__(self, title=None, text=None):
        if title == None or text == None:
            self.__title = None
            self.__body = None
            self.__value = None
            self.__hash = None
        else:
            self.__title = title
            self.__text = text
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
            return self.__value