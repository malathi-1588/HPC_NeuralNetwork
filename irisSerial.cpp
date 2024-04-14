#include <iostream>
#include <random>
#include <cmath>
#include <omp.h>
using namespace std;

float* getColumnwiseSum(float** matrix, int row, int col){
    float* result = new float[col]();
    for (int i=0; i<row; i++){
        for(int j=0; j<col; j++){
            result[j] += matrix[i][j];
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

float* getMeanVector(float* vec, int len, int denominator){
    float* result = new float[len];
    for (int i=0; i<len; i++){
        result[i] = vec[i] / denominator;
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

float** getData(float x_src[][4], float y_src[][3], int row, int col){
    float** data = new float*[row];
    for (int i=0; i<row; i++){
            data[i] = new float[col];
    }

    if (col == 4){
    	 
        for (int i=0; i<row; i++){
            for (int j=0; j<col; j++){
                data[i][j] = x_src[i][j];
            }
        }
    }
    else{
    	
        for (int i=0; i<row; i++){
            for (int j=0; j<col; j++){
                data[i][j] = y_src[i][j];
            }
        }
    }
    return data;
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
    int correct = 0;
    int total = rows * cols; 
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (y_true[i][j] == y_pred[i][j]) {
                correct++; 
            }
        }
    }
    
    return static_cast<double>(correct) / total;
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

float** getDivideMatrix(float** matrix, int row, int col, int scalar){
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
                z_activated = getSoftmax(z, row, wCol);
            }
            else{
                z_activated = getRelu(z, row, wCol);
            }
        }
};

void displayMatrix(float** matrix, int rows, int columns){
    for (int i=0; i<rows; i++){
        for (int j=0; j<columns; j++){
            cout << matrix[i][j] << " ";
        }
        cout << "\n";
    }
}

int main(){
    float x[150][4] = {
        {5.1, 3.5, 1.4, 0.2},{4.9, 3.0, 1.4, 0.2},{4.7, 3.2, 1.3, 0.2},{4.6, 3.1, 1.5, 0.2},{5.0, 3.6, 1.4, 0.2},
        {5.4, 3.9, 1.7, 0.4},{4.6, 3.4, 1.4, 0.3},{5.0, 3.4, 1.5, 0.2},{4.4, 2.9, 1.4, 0.2},{4.9, 3.1, 1.5, 0.1},
        {5.4, 3.7, 1.5, 0.2},{4.8, 3.4, 1.6, 0.2},{4.8, 3.0, 1.4, 0.1},{4.3, 3.0, 1.1, 0.1},{5.8, 4.0, 1.2, 0.2},
        {5.7, 4.4, 1.5, 0.4},{5.4, 3.9, 1.3, 0.4},{5.1, 3.5, 1.4, 0.3},{5.7, 3.8, 1.7, 0.3},{5.1, 3.8, 1.5, 0.3},
        {5.4, 3.4, 1.7, 0.2},{5.1, 3.7, 1.5, 0.4},{4.6, 3.6, 1.0, 0.2},{5.1, 3.3, 1.7, 0.5},{4.8, 3.4, 1.9, 0.2},
        {5.0, 3.0, 1.6, 0.2},{5.0, 3.4, 1.6, 0.4},{5.2, 3.5, 1.5, 0.2},{5.2, 3.4, 1.4, 0.2},{4.7, 3.2, 1.6, 0.2},
        {4.8, 3.1, 1.6, 0.2},{5.4, 3.4, 1.5, 0.4},{5.2, 4.1, 1.5, 0.1},{5.5, 4.2, 1.4, 0.2},{4.9, 3.1, 1.5, 0.1},
        {5.0, 3.2, 1.2, 0.2},{5.5, 3.5, 1.3, 0.2},{4.9, 3.1, 1.5, 0.1},{4.4, 3.0, 1.3, 0.2},{5.1, 3.4, 1.5, 0.2},
        {5.0, 3.5, 1.3, 0.3},{4.5, 2.3, 1.3, 0.3},{4.4, 3.2, 1.3, 0.2},{5.0, 3.5, 1.6, 0.6},{5.1, 3.8, 1.9, 0.4},
        {4.8, 3.0, 1.4, 0.3},{5.1, 3.8, 1.6, 0.2},{4.6, 3.2, 1.4, 0.2},{5.3, 3.7, 1.5, 0.2},{5.0, 3.3, 1.4, 0.2},
        {7.0, 3.2, 4.7, 1.4},{6.4, 3.2, 4.5, 1.5},{6.9, 3.1, 4.9, 1.5},{5.5, 2.3, 4.0, 1.3},{6.5, 2.8, 4.6, 1.5},
        {5.7, 2.8, 4.5, 1.3},{6.3, 3.3, 4.7, 1.6},{4.9, 2.4, 3.3, 1.0},{6.6, 2.9, 4.6, 1.3},{5.2, 2.7, 3.9, 1.4},
        {5.0, 2.0, 3.5, 1.0},{5.9, 3.0, 4.2, 1.5},{6.0, 2.2, 4.0, 1.0},{6.1, 2.9, 4.7, 1.4},{5.6, 2.9, 3.6, 1.3},
        {6.7, 3.1, 4.4, 1.4},{5.6, 3.0, 4.5, 1.5},{5.8, 2.7, 4.1, 1.0},{6.2, 2.2, 4.5, 1.5},{5.6, 2.5, 3.9, 1.1},
        {5.9, 3.2, 4.8, 1.8},{6.1, 2.8, 4.0, 1.3},{6.3, 2.5, 4.9, 1.5},{6.1, 2.8, 4.7, 1.2},{6.4, 2.9, 4.3, 1.3},
        {6.6, 3.0, 4.4, 1.4},{6.8, 2.8, 4.8, 1.4},{6.7, 3.0, 5.0, 1.7},{6.0, 2.9, 4.5, 1.5},{5.7, 2.6, 3.5, 1.0},
        {5.5, 2.4, 3.8, 1.1},{5.5, 2.4, 3.7, 1.0},{5.8, 2.7, 3.9, 1.2},{6.0, 2.7, 5.1, 1.6},{5.4, 3.0, 4.5, 1.5},
        {6.0, 3.4, 4.5, 1.6},{6.7, 3.1, 4.7, 1.5},{6.3, 2.3, 4.4, 1.3},{5.6, 3.0, 4.1, 1.3},{5.5, 2.5, 4.0, 1.3},
        {5.5, 2.6, 4.4, 1.2},{6.1, 3.0, 4.6, 1.4},{5.8, 2.6, 4.0, 1.2},{5.0, 2.3, 3.3, 1.0},{5.6, 2.7, 4.2, 1.3},
        {5.7, 3.0, 4.2, 1.2},{5.7, 2.9, 4.2, 1.3},{6.2, 2.9, 4.3, 1.3},{5.1, 2.5, 3.0, 1.1},{5.7, 2.8, 4.1, 1.3},
        {6.3, 3.3, 6.0, 2.5},{5.8, 2.7, 5.1, 1.9},{7.1, 3.0, 5.9, 2.1},{6.3, 2.9, 5.6, 1.8},{6.5, 3.0, 5.8, 2.2},
        {7.6, 3.0, 6.6, 2.1},{4.9, 2.5, 4.5, 1.7},{7.3, 2.9, 6.3, 1.8},{6.7, 2.5, 5.8, 1.8},{7.2, 3.6, 6.1, 2.5},
        {6.5, 3.2, 5.1, 2.0},{6.4, 2.7, 5.3, 1.9},{6.8, 3.0, 5.5, 2.1},{5.7, 2.5, 5.0, 2.0},{5.8, 2.8, 5.1, 2.4},
        {6.4, 3.2, 5.3, 2.3},{6.5, 3.0, 5.5, 1.8},{7.7, 3.8, 6.7, 2.2},{7.7, 2.6, 6.9, 2.3},{6.0, 2.2, 5.0, 1.5},
        {6.9, 3.2, 5.7, 2.3},{5.6, 2.8, 4.9, 2.0},{7.7, 2.8, 6.7, 2.0},{6.3, 2.7, 4.9, 1.8},{6.7, 3.3, 5.7, 2.1},
        {7.2, 3.2, 6.0, 1.8},{6.2, 2.8, 4.8, 1.8},{6.1, 3.0, 4.9, 1.8},{6.4, 2.8, 5.6, 2.1},{7.2, 3.0, 5.8, 1.6},
        {7.4, 2.8, 6.1, 1.9},{7.9, 3.8, 6.4, 2.0},{6.4, 2.8, 5.6, 2.2},{6.3, 2.8, 5.1, 1.5},{6.1, 2.6, 5.6, 1.4},
        {7.7, 3.0, 6.1, 2.3},{6.3, 3.4, 5.6, 2.4},{6.4, 3.1, 5.5, 1.8},{6.0, 3.0, 4.8, 1.8},{6.9, 3.1, 5.4, 2.1},
        {6.7, 3.1, 5.6, 2.4},{6.9, 3.1, 5.1, 2.3},{5.8, 2.7, 5.1, 1.9},{6.8, 3.2, 5.9, 2.3},{6.7, 3.3, 5.7, 2.5},
        {6.7, 3.0, 5.2, 2.3},{6.3, 2.5, 5.0, 1.9},{6.5, 3.0, 5.2, 2.0},{6.2, 3.4, 5.4, 2.3},{5.9, 3.0, 5.1, 1.8}
    };
    float y[150][3] = {
        {1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},
        {1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},
        {1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},
        {1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},
        {1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},
        {0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},
        {0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},
        {0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},
        {0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},
        {0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0}, 
        {0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1},
        {0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1},
        {0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1},
        {0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1},
        {0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1} 
    };
    
    double start_time = omp_get_wtime(); // Start time

    float** x_data = getData(x, y, 150, 4);
    float** y_data = getData(x, y, 150, 3);
    float** x_data_scaled = standardization(x_data, 150, 4);
    Layer l1;
    Layer l2;
    l1.setWeight(5, 32);
    l2.setWeight(33, 3);

    for(int i=0; i<100; i++){
        l1.forward(x_data_scaled, 150, 4, 0);
        l2.forward(l1.z_activated, 150, 32, 1);
        
        float** y_pred = classification(l2.z_activated, 150, 3);

        float mse = accuracy(y_data, y_pred, 150, 3);
            cout << " accuracy: " << i <<": "<< mse << "\n";

        float** lossDerivative = getDifferenceMatrix(y_data, y_pred, 150, 3, 1.0f);
        float** dw2 = getDotProduct(getTransposeMatrix(l2.a, 150, 33), lossDerivative,33, 150, 150, 3);
        float** w2_without_dummy_row = removeDummyRow(l2.w, 33, 3);
        float** da = getDotProduct(lossDerivative, getTransposeMatrix(w2_without_dummy_row, 32, 10), 150, 3, 3, 32);
        float** z1_drelu = getReluDerivative(l1.z, 150, 32);
        float** element_wise_product = getElementwiseMatrixMult(da, z1_drelu, 150, 32);
        float** dw1 = getDotProduct(getTransposeMatrix(l1.a, 150, 5), element_wise_product, 5, 150, 150, 32);

        float** updated_w1 = addMatrix(l1.w, dw1, 5, 32, 0.01);
        float** updated_w2 = addMatrix(l2.w, dw2, 33,3, 0.01);
        l1.w = updated_w1;
        l2.w = updated_w2;
    }
    double end_time = omp_get_wtime(); // End time
    double execution_time = end_time - start_time; // Execution time in seconds

    std::cout << "Execution time: " << execution_time << " seconds\n";
}



