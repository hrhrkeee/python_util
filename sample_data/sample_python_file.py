import time
class Exec_Timer:
    def __init__(self, output=True, desc:str=""):
        if len(desc)>0: desc = desc + ":\t"
        self.desc = desc
        self.output = output
    def __call__(self):    
        return float(time.time()-self.start)
    def __enter__(self):    
        self.start = time.time()
        return self
    def __exit__(self,*args): 
        if self.output:
            print ("{}elapsed_time:{:.06f}".format(self.desc, time.time()-self.start)+"[sec]")
        return True

if __name__ == '__main__':
    with Exec_Timer():
        time.sleep(1)
    with Exec_Timer(desc="timer"):
        time.sleep(2)

    with Exec_Timer(output=False) as t:
        print(int(t()))
        time.sleep(1)
        print(int(t()))
        time.sleep(2)
        print(int(t()*1000))