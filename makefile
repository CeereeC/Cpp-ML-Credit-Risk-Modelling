all: ml-app.o

ml-app.o: build main.o des.o gen.o eval.o
	g++ -std=c++14 -o ml-app.o build/*.o -larmadillo -lpthread

build:
	mkdir build
main.o:
	g++ -c -std=c++14 -o build/main.o main.cpp
des.o: 
	g++ -c -std=c++14 -o build/des.o deserializer/PredictRequestDeserializer.cpp  
gen.o:
	g++ -c -std=c++14 -o build/gen.o generator/ModelGenerator.cpp  
eval.o:
	g++ -c -std=c++14 -o build/eval.o eval/ModelEvaluator.cpp 

link:
	g++ -std=c++14 -o ml-app.o build/*.o -larmadillo -lpthread 
clean:
	rm build/des.o build/gen.o build/eval.o build/ml-app.o
