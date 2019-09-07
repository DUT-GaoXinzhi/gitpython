import socket
import threading
import struct
import cv2
import time
import os
# 这里全部是服务端的程序
class webCamera: 
    def __init__(self, resolution = (640, 480), host = ("", 7999)):      
        self.resolution = resolution    
        self.host = host
        self.setSocket(self.host)    
        self.img_quality = 15

    # 设置分辨率
    def setImageResolution(self, resolution):      
        self.resolution = resolution  
    # 设置端口号
    def setHost(self, host):      
        self.host = host
    # 设置对象的socket
    def setSocket(self, host):      
        self.socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        # 绑定端口号            
        self.socket.bind(self.host)
        # 开始监听   
        self.socket.listen(5)
        print("Server running on port:%d" % host[1])
    # 接收设置
    def recv_config(self,client):
        # 将收到的消息解压缩    
        info = struct.unpack("lhh",client.recv(8)) 
        # 如果接收到的消息的第一位大于911
        # 说明收到的消息时正确的按照格式提取相应的消息
        if info[0]>911:        #print(info[0])        
            self.img_quality=int(info[0])-911              
            self.resolution=list(self.resolution)        
            self.resolution[0]=info[1]        
            self.resolution[1]=info[2]        
            self.resolution=tuple(self.resolution)                
            return 1    
        else :        
            return 0
    # 连接过程的处理
    def _processConnection(self, client,addr):
        if(self.recv_config(client)==0):        
            return    
        camera = cv2.VideoCapture(0)    
        encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),self.img_quality]             
        f = open("video_info.txt", 'a+')    
        print("Got connection from %s:%d" % (addr[0], addr[1]),file=f)
        print("像素为:%d * %d"%(self.resolution[0],self.resolution[1]),file=f)    
        print ("打开摄像头成功",file=f)    
        print("连接开始时间:%s"%time.strftime("%Y-%m-%d %H:%M:%S", 
                time.localtime(time.time())),file=f)    
        f.close()    
        while(1):        
            time.sleep(0.13)        
            (grabbed, self.img) = camera.read()        
            self.img  = cv2.resize(self.img,self.resolution)             
            result, imgencode = cv2.imencode('.jpg',self.img,encode_param)        
            img_code = numpytest.array(imgencode)
            self.imgdata  = img_code.tostring()                   
            try:                    
                client.send(struct.pack("lhh",len(self.imgdata),
                        self.resolution[0],self.resolution[1])+self.imgdata)        
            except:            
                f = open("video_info.txt", 'a+')            
                print("%s:%d disconnected!" % (addr[0], addr[1]),file=f)
                print("连接结束时间:%s"%time.strftime("%Y-%m-%d %H:%M:%S", 
                    time.localtime(time.time())),file=f)      
                print("****************************************",file=f)
                camera.release()            
                f.close()            
                return
    def run(self):      
        while(1):        
            client,addr = self.socket.accept()
            clientThread = threading.Thread(target = self._processConnection, 
                args = (client, addr, ))  #有客户端连接时产生新的线程进行处理
            clientThread.start()
    def main(self):
        self.run()
if __name__ == "__main__":
    test = webCamera()
    test.main()