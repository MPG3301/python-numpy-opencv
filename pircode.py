import RPi.GPIO as GPIO
import time

sensor = 16
buzzer = 18

GPIO.setmode(GPIO.BOARD)
GPIO.setup(sensor,GPIO.IN)
GPIO.setup(buzzer,GPIO.OUT)

GPIO.output(buzzer,False)
print "Initialzing PIR Sensor......"
time.sleep(12)
print "PIR Ready..."
print " "

try: 
   while True:
      if GPIO.input(sensor):
          GPIO.output(buzzer,True)
          print "Motion Detected"
          while GPIO.input(sensor):
              time.sleep(0.2)
      else:
          GPIO.output(buzzer,False)


except KeyboardInterrupt:
    GPIO.cleanup()
