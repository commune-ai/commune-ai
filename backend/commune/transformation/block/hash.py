
class String2IntHash(object):

    @staticmethod
    def transform(x):
        return int(''.join(map(lambda x: '%.3d' % ord(x), x)))
    @staticmethod
    def inverse(x):
        s = str(x)
        if len(s) % 3 != 0:
            s = "0" + s
        return ''.join([chr(int(s[i:i+3])) for i in range(0, len(s), 3)])

    @staticmethod
    def circle(x):
        return String2IntHash.inverse(String2IntHash.transform(x))