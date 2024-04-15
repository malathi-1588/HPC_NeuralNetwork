#include <iostream>
#include <random>
#include <cmath>
#include<omp.h>
using namespace std;

float** getIndependentData(int row, int col){

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(100.0, 5.0);

    float** x_data = new float*[row];
    for (int i=0; i<row; i++){
        x_data[i] = new float[col];
    }
    for (int i=0; i<row; i++){
        for (int j=0; j<col; j++){
            x_data[i][j] = dis(gen);
        }
    }
    return x_data;
}

float** getDependentData(float** x_data, int row, int col){
    float** y_data = new float*[row];
    for (int i=0; i<row; i++){
        y_data[i] = new float[col];
    }

    for (int i=0; i<row; i++){
        for (int j=0; j<col; j++){
            y_data[i][j] = (x_data[i][j] * 2) + 5;
        }
    }
    return y_data;
}

void displayMatrix(float** matrix, int rows, int columns){
    for (int i=0; i<rows; i++){
        for (int j=0; j<columns; j++){
            cout << matrix[i][j] << " ";
        }
        cout << "\n";
    }
}

float* getColumnwiseSum(float** matrix, int row, int col){
    float* result = new float[col]();
    for (int i=0; i<row; i++){
        for(int j=0; j<col; j++){
            result[j] += matrix[i][j];
        }
    }
    return result;
}

float* getMeanVector(float* vec, int len, int denominator){
    float* result = new float[len];
    for (int i=0; i<len; i++){
        result[i] = vec[i] / denominator;
    }
    return result;
}

float** getDeviationMean(float** matrix, int row, int col, float* means, int len){
    float** result = new float*[row];
    for (int i=0; i<row; i++){
        result[i] = new float[col];
    } 
     
    for (int i=0; i<row; i++){
        for (int j=0; j<col; j++){
            result[i][j] = matrix[i][j] - means[j];
        }
    }
    return result;
}

float** getSquareMatrix(float** matrix, int row, int col){
    float** result = new float*[row];
    for (int i=0; i<row; i++){
        result[i] = new float[col];
    } 
     
    for (int i=0; i<row; i++){
        for(int j=0; j<col; j++){
            result[i][j] = (matrix[i][j] * matrix[i][j]);
        }
    }
    return result;
}

float* getSquareRootVector(float* vec, int len){
    float* result = new float[len];
    for (int i=0; i<len; i++){
        result[i] = sqrt(vec[i]);
    }
    return result;
}

float** standardization(float** matrix, int row, int col){

    float* col_sums = getColumnwiseSum(matrix, row, col);
    float* means = getMeanVector(col_sums, col, row);
    float** diff = getDeviationMean(matrix, row, col, means, col);
    float** squared_diff = getSquareMatrix(diff, row, col);
    float* col_sums_sq_diff = getColumnwiseSum(squared_diff, row, col);
    float* variances = getMeanVector(col_sums_sq_diff, col, row);
    float* std_devs = getSquareRootVector(variances, col);

    float** result = new float*[row];
    for (int i=0; i<row; i++){
        result[i] = new float[col];
    }
    
	
    for (int i=0; i<row; i++){
        for (int j=0; j<col; j++){
            result[i][j] = ((matrix[i][j] - means[j]) / std_devs[j]);
        }
    }
    return result;
}
float** initializeWeightMatrix(int rows, int columns){

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(0.0, 0.1);

    float** weight_matrix = new float*[rows];
    for (int i = 0; i < rows; i++){
        weight_matrix[i] = new float[columns];
    }

	
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < columns; j++){
            weight_matrix[i][j] = distribution(gen);
        }
    }
    return weight_matrix;
}
float** addingDummyColumn(float** matrix, int rows, int columns){
    for (int i = 0; i < rows; i++){
        matrix[i][columns] = 1.0f;
    }
    return matrix;
}
float** getDotProduct(float** weights, float** inputs, int rows1, int cols1, int rows2, int cols2){
    float** result = new float*[rows1];
    for (int i = 0; i < rows1; ++i) {
        result[i] = new float[cols2]();
    }

	
    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols2; ++j) {
            for (int k = 0; k < cols1; ++k) {
                result[i][j] += weights[i][k] * inputs[k][j]; 
            }
        }
    }
    return result;
}
float** getExponential(float** matrix, int row, int col){
    float** result = new float*[row];
    for (int i=0; i<row; i++){
        result[i] = new float[col];
    }
    
    for (int i=0; i<row; i++){
        for(int j=0; j<col; j++){
            result[i][j] = exp(matrix[i][j]);
        }
    }
    return result;
}
float* getRowwiseSum(float** matrix, int row, int col){
    float* result = new float[row]();
    for (int i=0; i<row; i++){
        for(int j=0; j<col; j++){
            result[i] += matrix[i][j];
        }
    }
    return result;
}
float** getSoftmax(float** matrix, int row, int col){
    float** result = new float*[row];
    for (int i=0; i<row; i++){
        result[i] = new float[col];
    }

    float** exponents = getExponential(matrix, row, col);
    float* sumOfExponents = getRowwiseSum(exponents, row, col);
    
    for (int i=0; i<row; i++){
        for (int j=0; j<col; j++){
            result[i][j] = (exponents[i][j] / sumOfExponents[i]);
        }
    }
    return result;
}
float** getRelu(float** matrix, int row, int col){
    float** result = new float*[row];
    for (int i=0; i<row; i++){
        result[i] = new float[col];
    }
	
    for (int i=0; i<row; i++){
        for (int j=0; j<col; j++){
            if (matrix[i][j] < 0){
                result[i][j] = 0;
            }
            else{
                result[i][j] = matrix[i][j];
            }
        }
    }
    return result;
}
int* getMaxIndex(float** matrix, int row, int col){
    int* result = new int[row];
    for (int i=0; i<row; i++){
        int temp = 0;
        float max = 0.0;
        for (int j=0; j<col; j++){
            if (matrix[i][j] >= max){
                max = matrix[i][j];
                temp = j;
            }
        }
        result[i] = temp;
    }
    return result;
}
float** classification(float** matrix, int row, int col){
    int* maxIdxVec = getMaxIndex(matrix, row, col);
    float** result = new float*[row];
    for (int i=0; i<row; i++){
        result[i] = new float[col]();
    }
    
    
    for (int i=0; i<row; i++){
        for (int j=0; j<col; j++){
            if (maxIdxVec[i] == j){
                result[i][j] += 1;
            }
        }
    }
    return result;
}
float accuracy(float** y_true, float** y_pred, int rows, int cols) {
    float sum_squared_errors = 0.0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; j++){
            float error = y_true[i][j] - y_pred[i][j];
            sum_squared_errors += error * error;
        }
    }
    return sum_squared_errors / rows;
}

float** getDifferenceMatrix(float** matrix1, float** matrix2, int rows, int columns, float scalar){
    float** difference = new float*[rows];
    for (int i=0; i<rows; i++){
        difference[i] = new float[columns];
    }

	
    for (int i=0; i<rows; i++){
        for (int j=0; j<columns; j++){
            difference[i][j] = matrix1[i][j] - (scalar * matrix2[i][j]);
        }
    }
    return difference;
}
float** getTransposeMatrix(float** weight_matrix, int rows, int columns){
    float** transpose = new float* [columns];
    for (int i = 0; i < columns; i++){
        transpose[i] = new float[rows];
    }

	
    for (int i = 0; i < columns; i++){
        for (int j = 0; j < rows; j++){
            transpose[i][j] = weight_matrix[j][i];
        }
    }
    return transpose;
}
float** getReluDerivative(float** matrix, int row, int col){
    float** result = new float*[row];
    for (int i=0; i<row; i++){
        result[i] = new float[col];
    }
	
    for (int i=0; i<row; i++){
        for (int j=0; j<col; j++){
            if (matrix[i][j] < 0){
                result[i][j] = 0;
            }
            else{
                result[i][j] = 1;
            }
        }
    }
    return result;
}
float** getElementwiseMatrixMult(float** matrix1, float** matrix2, int row, int col){
    float** result = new float*[row];
    for (int i=0; i<row; i++){
        result[i] = new float[col];
    }
    
    for (int i=0; i<row; i++){
        for (int j=0; j<col; j++){
            result[i][j] = matrix1[i][j] * matrix2[i][j];
        }
    }
    return result;
}

float** addMatrix(float** matrix1, float** matrix2, int rows, int columns, float scalar){
    float** add = new float*[rows];
    for (int i=0; i<rows; i++){
        add[i] = new float[columns];
    }
    
    for (int i=0; i<rows; i++){
        for (int j=0; j<columns; j++){
            add[i][j] = matrix1[i][j] + (scalar * matrix2[i][j]);
        }
    }
    return add;
}
float** removeDummyRow(float** matrix, int row, int col){
    float** result = new float*[row-1];
    for (int i=0; i<(row-1); i++){
        result[i] = new float[col];
    }
    
    for (int i=0; i<(row-1); i++){
        for (int j=0; j<col; j++){
            result[i][j] = matrix[i][j];
        }
    }
    return result;
}
float** divideMatrix(float** matrix, int row, int col, int scalar){
    float** result = new float*[row];
    for (int i=0; i<row; i++){
        result[i] = new float[col];
    }

    for (int i=0; i<row; i++){
        for (int j=0; j<col; j++){
            result[i][j] = matrix[i][j] / scalar;
        }
    }
    return result;
}
class Layer
{
    public:
        int wRow;
        int wCol;
        float** w;
        float** a;
        float** z;
        float** z_activated;
        void setWeight(int w_row, int w_col){
            wRow = w_row;
            wCol = w_col;
            w = initializeWeightMatrix(w_row, w_col);
        }
        void forward(float** input, int row, int col, int finalLayerFlag){
            a = addingDummyColumn(input, row, col);
            z = getDotProduct(a, w, row, col+1, wRow, wCol);
            if (finalLayerFlag){
               z_activated = getRelu(z, row, wCol);
            }
            else{
                z_activated = z;
            }
        }
};

int main(){
	
	double start_time = omp_get_wtime(); // Start time
	
    int row = 600;
    int col = 1;
    int units = 300;
    float** x_data = getIndependentData(row, col);
    float** y_data = getDependentData(x_data, row, col);
    float** x_data_scaled = standardization(x_data, row, col);
    Layer l1;
    Layer l2;
    l1.setWeight(col+1, units);
    l2.setWeight(units+1, col);

    for(int i=0; i<1100; i++){
        l1.forward(x_data_scaled, row, col, 1);
        l2.forward(l1.z_activated, row, units, 0);
        
        //float** y_pred = classification(l2.z_activated, row, col);

        float mse = accuracy(y_data, l2.z, row, col);
            cout << " mse: " << i <<": "<< mse << "\n";

        float** difference = getDifferenceMatrix(y_data, l2.z, row, col, 1.0f);
        float** lossDerivative = divideMatrix(difference, row, col, 2*row);
        float** dw2 = getDotProduct(getTransposeMatrix(l2.a, row, units+1), lossDerivative,units+1, row, row, col);
        float** w2_without_dummy_row = removeDummyRow(l2.w, units+1, col);
        float** da = getDotProduct(lossDerivative, getTransposeMatrix(w2_without_dummy_row, units, col), row, col, col, units);
        float** z1_drelu = getReluDerivative(l1.z, row, units);
        float** element_wise_product = getElementwiseMatrixMult(da, z1_drelu, row, units);
        float** dw1 = getDotProduct(getTransposeMatrix(l1.a, row, col+1), element_wise_product, col+1, row, row, units);

        float** updated_w1 = addMatrix(l1.w, dw1, col+1, units, 0.01);
        float** updated_w2 = addMatrix(l2.w, dw2, units+1, col, 0.01);
        l1.w = updated_w1;
        l2.w = updated_w2;
    }
    double end_time = omp_get_wtime(); // End time
    double execution_time = end_time - start_time; // Execution time in seconds

    std::cout << "Serial Execution time: " << execution_time << " seconds\n";
}