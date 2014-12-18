#include <iostream>
#include <vector>
#include <fstream>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#define	INPUT_LAYER	0
#define	HIDDEN_LAYER	1
#define	OUTPUT_LAYER	2

class NeuronLayer{
public:
	int TypeOfLayer;
	int NumOfNeuron;
	NeuronLayer *PointerOfNext;
	NeuronLayer *PointerOfPrev;
	Eigen::MatrixXd WeightMatrix;
	Eigen::VectorXd OutputVector;
	Eigen::VectorXd	DeltaVector;
	//Constructer
	NeuronLayer(){
		this->TypeOfLayer = 0;
		this->NumOfNeuron = 0;
		this->PointerOfNext = NULL;
		this->PointerOfPrev = NULL;
		return;
	}
	void Init(int _type,int _num,NeuronLayer *_pp,NeuronLayer *_np){
		this->TypeOfLayer = _type;
		if(_type != OUTPUT_LAYER) _num+=1;
		this->NumOfNeuron = _num;
		this->PointerOfNext = _np;
		this->PointerOfPrev = _pp;
		OutputVector = Eigen::VectorXd::Zero( _num );
		DeltaVector = Eigen::VectorXd::Zero( _num );
		if(_type != INPUT_LAYER)
			WeightMatrix = Eigen::MatrixXd::Random( _num , _pp->NumOfNeuron );
		//std::cout << "(this)" << this << std::endl;
		return;
	}
	~NeuronLayer(){
				
	}


	//Forward Compute
	void Compute(Eigen::VectorXd _input){
		this->OutputVector(0) = 1.0;
		for(int i=0;i<(_input.rows());i++) this->OutputVector(i+1) = _input(i);
		Compute();
	}
	void Compute(){
		if( this->TypeOfLayer != INPUT_LAYER ){
			OutputVector = WeightMatrix * this->PointerOfPrev->OutputVector;
			int this_num = this->NumOfNeuron;
			for(int i=0;i<this_num;i++)
				OutputVector(i) =  1.0 / ( 1.0 + exp(-OutputVector(i)) );
			if(this->TypeOfLayer != OUTPUT_LAYER) OutputVector(0) = 1.0;
		}
		if( this->TypeOfLayer != OUTPUT_LAYER )
			this->PointerOfNext->Compute();
		return;
	}

	//BackPropagation
	double BackPropagation(Eigen::VectorXd _teach,double alpha){
		this->DeltaVector = -1.0 * ( _teach - this->OutputVector ).cwiseProduct( this->OutputVector )
		.cwiseProduct( Eigen::VectorXd::Ones(this->NumOfNeuron) - this->OutputVector );
		this->WeightMatrix += -alpha * ( Eigen::MatrixXd(this->WeightMatrix).colwise() = this->DeltaVector )
		.cwiseProduct(Eigen::MatrixXd(this->WeightMatrix).rowwise() = this->PointerOfPrev->OutputVector.transpose());
		this->PointerOfPrev->BackPropagation( alpha );
		return (_teach - this->OutputVector).squaredNorm() * 0.5;
	}
	void BackPropagation(double alpha){
		this->DeltaVector = (this->PointerOfNext->WeightMatrix.transpose() * this->PointerOfNext->DeltaVector)
		.cwiseProduct( this->OutputVector ).cwiseProduct(Eigen::VectorXd::Ones(this->NumOfNeuron) - this->OutputVector);
		this->WeightMatrix += -alpha * ( Eigen::MatrixXd(this->WeightMatrix).colwise() = this->DeltaVector )
		.cwiseProduct(Eigen::MatrixXd(this->WeightMatrix).rowwise() = this->PointerOfPrev->OutputVector.transpose());
		if(this->PointerOfPrev->TypeOfLayer != INPUT_LAYER) this->PointerOfPrev->BackPropagation( alpha );
		return;
	}

};

class NeuralNetwork{
private:
public:
	NeuronLayer *layer;
	int NumOfLayer;
	int offset;

	NeuralNetwork(){
		this->layer = NULL;
		this->NumOfLayer = 0;
		this->offset = 0;
	}
	~NeuralNetwork(){
		delete[] layer;
	}

	int Init(int num){
		this->NumOfLayer = num;
		layer = new NeuronLayer[num];
		return 0;
	}

	int Add(int num){
		if( this->offset < this->NumOfLayer ){
			std::cout << "(Add) " << this->offset << std::endl;
			if(this->offset==0)
				layer[this->offset].Init( INPUT_LAYER , num , NULL , &layer[this->offset+1]);
			else if(this->offset==this->NumOfLayer-1)
				layer[this->offset].Init( HIDDEN_LAYER , num , &layer[this->offset-1] , &layer[this->offset+1]);
			else 
				layer[this->offset].Init( OUTPUT_LAYER , num , &layer[this->offset-1] , NULL );
			this->offset++;
			return 0;
		}
		std::cout << "(Err) overflow" << std::endl;
		return -1;
	}

	int Add(int id,int type,int num){
		NeuronLayer *p1,*p2;
		p1 = &layer[id-1];
		p2 = &layer[id+1];
		if(id==0) p1 = NULL;
		if(id==this->NumOfLayer-1) p2 = NULL;
		layer[id].Init( type , num , p1 , p2 );

		return 0;
	}

	int Show(){
		for(int i=0;i<(this->NumOfLayer);i++){
			if(layer[i].NumOfNeuron!=0){
				std::cout << "[" << i << "/" << this->NumOfLayer << "] (" << &layer[i] << ") Prev=" << layer[i].PointerOfPrev;
				std::cout << " Next=" << layer[i].PointerOfNext << std::endl; 
			}else{
				std::cout << "[" << i << "/" << this->NumOfLayer << "] " << std::endl;
			}
		}
		return 0;
	}


	Eigen::VectorXd Compute( Eigen::VectorXd _input ){
		layer[0].Compute( _input );
		return layer[this->NumOfLayer-1].OutputVector;
	}
	double BackPropagation( Eigen::VectorXd _teach , double _rate ){
		return layer[this->NumOfLayer-1].BackPropagation( _teach , _rate );
	}
	

	
	Eigen::MatrixXd ComputeAuto( Eigen::MatrixXd _input ){
		Eigen::MatrixXd res( _input.rows() , layer[this->NumOfLayer-1].NumOfNeuron );
		for(int i=0;i<_input.rows();i++){
			layer[0].Compute( _input.row(i) );
			res.row(i) = layer[this->NumOfLayer-1].OutputVector;
		}
		return res;
	}
	
	double BackPropagationAuto( Eigen::MatrixXd _a , Eigen::MatrixXd _b , double rate , double err){
		for(int iter=0;;iter++){
			double e = 0.0;
			//Try 
			for(int i=0;i<_a.rows();i++){
				Compute( _a.row(i) );
				e+=BackPropagation( _b.row(i) , rate );
			}
			//std::cout << "e=" << e << std::endl;
			if( e < err ){
				break;
			}
		}
		return 0.0;
	}
	

	bool SaveToFile(const char* fileName){
		std::vector<double> testVec;
		testVec.push_back( this->NumOfLayer );
		for(int i=1;i<this->NumOfLayer;i++){
			testVec.push_back( layer[i].NumOfNeuron );
			testVec.push_back( layer[i-1].NumOfNeuron );
			for(int j=0;j<layer[i].NumOfNeuron;j++){
				for(int k=0;k<layer[i-1].NumOfNeuron;k++){
					testVec.push_back( layer[i].WeightMatrix(j,k) );
				}
			}
		}
		std::ofstream ofs(fileName, std::ios::binary);
		if (ofs.fail()) return false;
		ofs.write(reinterpret_cast<const char*>(&testVec[0]), sizeof(double) * testVec.size());
		ofs.flush();
		if (ofs.fail()) return false; else return true;	
	}

	bool LoadFromFile(const char* fileName){
		std::ifstream ifs(fileName, std::ios::binary);
		if (ifs.fail()) return false;
		const size_t fileSize = static_cast<size_t>(ifs.seekg(0, std::ios::end).tellg());
		ifs.seekg(0, std::ios::beg);
		if (fileSize > 0 && fileSize % sizeof(double) == 0){
			std::vector<double> testVec(fileSize / sizeof(double));
			ifs.read(reinterpret_cast<char*>(&testVec[0]), fileSize);
			int offset = 0;
			int LayerNum = testVec[ offset++ ];
			for(int i=1;i<LayerNum;i++){
				int row = testVec[ offset++ ];
				int col = testVec[ offset++ ];
				for(int j=0;j<row;j++) for(int k=0;k<col;k++)
					layer[i].WeightMatrix(j,k) = testVec[ offset++ ];	
			}
		}else return false;
		return true;
	}


};

	
	/* File */
	template<typename T_n> void SaveEigen(const char* fileName , T_n _mat ){
		std::vector<double> testVec;
		std::vector<double> testVec2;
		testVec.push_back( _mat.rows() );
		testVec.push_back( _mat.cols() );
		for(int i=0;i<_mat.cols();i++)
				for(int j=0;j<_mat.rows();j++)
					testVec.push_back( _mat(j,i) );
		std::ofstream ofs(fileName, std::ios::binary);
		ofs.write(reinterpret_cast<const char*>(&testVec[0]), sizeof(double) * testVec.size());
		ofs.flush();
		ofs.close();
		return;
	}

	template<typename T_n> T_n LoadEigen(const char* fileName){
		std::ifstream ifs(fileName, std::ios::binary);
		const size_t fileSize = static_cast<size_t>(ifs.seekg(0, std::ios::end).tellg());
		ifs.seekg(0, std::ios::beg);
		std::vector<double> testVec(fileSize / sizeof(double));
		ifs.read(reinterpret_cast<char*>(&testVec[0]), fileSize);
		ifs.close();
		int offset = 0;
		int row = testVec[ offset++ ];
		int col = testVec[ offset++ ];
		//std::cout << "row=" << row << std::endl;
		//std::cout << "col=" << col << std::endl;
		return Eigen::Map<T_n>(&testVec[offset],row,col);
	}
	

int learn(){

	NeuralNetwork nn1;
	
	nn1.Init(3);
	nn1.Add(0,INPUT_LAYER,25);
	nn1.Add(1,HIDDEN_LAYER,20);
	nn1.Add(2,OUTPUT_LAYER,10);
	nn1.Show();

	Eigen::MatrixXd Input(10,25);
	Eigen::MatrixXd Output(10,10);

	Input = LoadEigen<Eigen::MatrixXd>("Input.dat");
	Output = LoadEigen<Eigen::MatrixXd>("Output.dat");

	if( nn1.LoadFromFile("nn1.dat") == false ){
		std::cout << "Learning..." << std::endl;
		nn1.BackPropagationAuto( Input , Output , 0.5  , 10E-3 );
		nn1.SaveToFile("nn1.dat");
	}else{
		std::cout << "Loaded." << std::endl;
	}

	std::cout << nn1.ComputeAuto( Input ) << std::endl;


	return 0;
}

int MakeData(){

	Eigen::MatrixXd Input(10,25);
	Eigen::MatrixXd Output = Eigen::MatrixXd::Identity(10,10);

	Input <<
	//0
	1,1,1,1,1,
	1,0,0,0,1,
	1,0,0,0,1,
	1,0,0,0,1,
	1,1,1,1,1,
	//1
	0,0,1,0,0,
	0,0,1,0,0,
	0,0,1,0,0,
	0,0,1,0,0,
	0,0,1,0,0,
	//2
	1,1,1,1,1,
	0,0,0,0,1,
	1,1,1,1,1,
	1,0,0,0,0,
	1,1,1,1,1,
	//3
	1,1,1,1,1,
	0,0,0,0,1,
	1,1,1,1,1,
	0,0,0,0,1,
	1,1,1,1,1,
	//4
	1,0,0,1,0,
	1,0,0,1,0,
	1,1,1,1,1,
	0,0,0,1,0,
	0,0,0,1,0,
	//5
	1,1,1,1,1,
	1,0,0,0,0,
	1,1,1,1,1,
	0,0,0,0,1,
	1,1,1,1,1,
	//6
	1,0,0,0,0,
	1,0,0,0,0,
	1,1,1,1,1,
	1,0,0,0,1,
	1,1,1,1,1,
	//7
	1,1,1,1,1,
	1,0,0,0,1,
	1,0,0,0,1,
	0,0,0,0,1,
	0,0,0,0,1,
	//8
	1,1,1,1,1,
	1,0,0,0,1,
	1,1,1,1,1,
	1,0,0,0,1,
	1,1,1,1,1,
	//9
	1,1,1,1,1,
	1,0,0,0,1,
	1,1,1,1,1,
	0,0,0,0,1,
	0,0,0,0,1;


	SaveEigen<Eigen::MatrixXd>("Input.dat",Input);
	SaveEigen<Eigen::MatrixXd>("Output.dat",Output);

	return 0;
}

int main(int argc,char *argv[]){

	MakeData();

	learn();

	return 0;
}

