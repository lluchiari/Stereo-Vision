PROG=open_image

all:
	@g++ ${PROG}.cpp -o ${PROG} -lopencv_core -lopencv_imgcodecs -lopencv_imgproc
 
compile:
	@g++ -c ${PROG}.cpp

run: all
	@./${PROG}

clean:
	@rm *.o ${PROG}


