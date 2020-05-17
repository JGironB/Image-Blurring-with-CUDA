//Librerias estándar de C
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <sys/time.h>
#include <string.h>

//Librerias de terceros usadas para manipulación de imágenes png, jpg, etc.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_library/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_library/stb_image_write.h"

//Struct de parámetros pasados a los hilos para ejecutar la convolución 
struct convolution_args {

    size_t sourcePixel;
    size_t endPixel;
    double** kernel;
    unsigned char* img;
    unsigned char* blurred_img;
    size_t channels;
    size_t blur_channels;
    size_t kernel_size;
    size_t width;
    size_t height;
    int thread_id;
    int n_threads;
};


//Función que realiza la convolución dados los parametros del rango de pixeles
void executeConvolution(struct convolution_args args){

    //Construcción de tamaño de imágen que se quiere producir
    size_t img_size = args.width * args.height;
    size_t blur_channels = args.blur_channels;
    size_t blurred_image_size = img_size * blur_channels;

    unsigned char* blurred_img = args.blurred_img;
    unsigned char* img = args.img;

    //Rango de pixeles entre los que se va a trabajar
    size_t sourcePixel = args.sourcePixel;
    size_t endPixel = args.endPixel;

    unsigned char* p = img + (sourcePixel*args.channels);
    unsigned char* b_p = blurred_img + (sourcePixel*blur_channels);

    size_t channels = args.channels;

    double valueRed;
    double valueGreen;
    double valueBlue;

    int current_pixel = sourcePixel;
    int target_pixel;
    int pixel_valueBlue;
    int pixel_valueGreen;
    int pixel_valueRed;

    int mid_size = args.kernel_size/2;

    size_t width = args.width;
    size_t height = args.height;

    double** kernel = args.kernel;

    //Loop que itera sobre el rango de pixeles dado como argumentos
    for(; p != img+((endPixel+1)*channels); p += channels, b_p += blur_channels, ++current_pixel) {
        
        valueRed = 0;
        valueGreen = 0;
        valueBlue = 0;

            //Recorrido por cada uno de los valores de la matriz del kernel
            for(int i = -mid_size; i <= mid_size; ++i){
                for(int j = -mid_size; j <= mid_size; ++j){
                    //Calculo de pixel de imágen original sobre el que queremos aplicar convolución
                    target_pixel = current_pixel + (i*width) + j;

                    //Extracción de cada uno de los tres canales del pixel identificado
                    pixel_valueRed = target_pixel < 0 || target_pixel > width*height-1 ? 1 : *(img+(target_pixel*channels)+0);
                    pixel_valueGreen = target_pixel < 0 || target_pixel > width*height-1 ? 1 : *(img+(target_pixel*channels)+1);
                    pixel_valueBlue = target_pixel < 0 || target_pixel > width*height-1 ? 1 : *(img+(target_pixel*channels)+2);

                    //Suma de valores multiplicados
                    valueRed += kernel[i+mid_size][j+mid_size] * pixel_valueRed;
                    valueGreen += kernel[i+mid_size][j+mid_size] * pixel_valueGreen;
                    valueBlue += kernel[i+mid_size][j+mid_size] * pixel_valueBlue;
                }
            } 

        //Asignación de canales modificados a la imágen con filtro
        *b_p =  (uint8_t)(valueRed);
        *(b_p + 1)=  (uint8_t)(valueGreen);
        *(b_p + 2)=  (uint8_t)(valueBlue);  
    }
}
  

//Función para generar el kernel gaussiano
void generate_kernel(int size, double** kernel) 
{   
    //Desviación estándar 
    double sigma = 15.0; 
    
    //Suma para normalizar el kernel después 
    double sum = 0.0; 
    
    int mid = size/2;

    //Generar kernel de tamaño size x size usando funcion de densidad de probabilidad de Gauss
    for (int x = -mid; x <= mid; x++) { 
        for (int y = -mid; y <= mid; y++) { 
            kernel[x + mid][y + mid] = exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma); 
            sum += kernel[x + mid][y + mid]; 
        } 
    } 
  
    //Normalización del kernel para un efecto mas limpio en el blurring
    for (int i = 0; i < size; ++i) 
        for (int j = 0; j < size; ++j) 
            kernel[i][j] /= sum; 
} 


//Función que asigna trabajo a cada uno de los hilos
void *assignWork(void *args) 
{ 
    struct convolution_args * my_args = (struct convolution_args *)args;

    size_t img_size = my_args->width * my_args->height;

    //Repartición de trabajo según blockwise
    size_t load_work = img_size/(my_args->n_threads);

    //Calculamos rango de pixeles a trabajar segun el id del pixel
    my_args->sourcePixel = my_args->thread_id * load_work;
    my_args->endPixel = (my_args->thread_id+1)* load_work - 1;

    printf("\nThread number %d executing from pixel %ld to %ld\n", my_args->thread_id, my_args->sourcePixel, my_args->endPixel);
    executeConvolution(*my_args);

    return NULL; 
}


int main(int argc, char* argv[]) {

    int width, height, channels;

    //Verificación de cantidad de argumentos correcta
    if(argc != 5) {
        perror("Cantidad de argumentos no es valida!");
        return EXIT_FAILURE;
    }

    //Cargamos la imagen obteniendo sus datos
    unsigned char* img = stbi_load(argv[1], &width, &height, &channels, 0);

    //Verificación de imágen válida
    if(img == NULL) {
        perror("Error cargando la imagen!\n");
        return EXIT_FAILURE;
    }

    printf("\nancho: %dpx, alto: %dpx, canales: %d\n", width, height, channels);

    int kernel_size;

    //Extracción de tamaño del kernel 
    kernel_size = atoi(argv[3]);

    //Tamaño de kernel debe ser impar
    if(kernel_size % 2 == 0){

        perror("Tamaño de kernel debe ser impar!\n");
        return EXIT_FAILURE;
    }

    size_t mid_size = kernel_size / 2 ;

    //Asignación dinámica de espacio para generar una matriz 
    double** kernel = (double**)malloc(sizeof(double*) * kernel_size);
    for(size_t i = 0; i < kernel_size; ++i) 
        kernel[i] = (double*)malloc(sizeof(double)*kernel_size);
    
    generate_kernel(kernel_size, kernel);

    printf("\nKernel gaussiano usado para el filtro: \n\n");
    for(size_t i = 0; i < kernel_size; ++i) {

        for(size_t j = 0; j < kernel_size; ++j) 
            printf("%f ", kernel[i][j]);
        
        printf("\n");
    }

    size_t blurred_image_size = width * height * 3;

    //Asignación de espacio para imágen con filtro aplicado
    unsigned char* blurred_img = (unsigned char*)malloc(sizeof(unsigned char) * blurred_image_size);

    int n_threads = atoi(argv[4]);

    struct convolution_args args[n_threads];

    pthread_t tid[n_threads];

    struct timeval start, end;

    //Calculo de tiempo antes de iniciar operaciones de convolución
	gettimeofday(&start, NULL);

    for (int i = 0; i < n_threads; i++) {

        args[i].img = img;
        args[i].blurred_img = blurred_img;
        args[i].width = width;
        args[i].height = height;
        args[i].channels = channels;
        args[i].blur_channels = 3;
        args[i].kernel = kernel;
        args[i].kernel_size = kernel_size;
        args[i].n_threads = n_threads;
        args[i].thread_id = i;

        //Lanzamiento de cada uno de los threads
        pthread_create(&tid[i], NULL, assignWork, (void *)&args[i]);
    } 

    for (int i = 0; i < n_threads; i++) 
        pthread_join(tid[i], NULL);

    //Calculo de tiempo después de aplicar convolución a todos los pixeles de la imágen
    gettimeofday(&end, NULL);
    
    //Escribimos la imágen en formato jpg con el filtro aplicado
    stbi_write_jpg(argv[2], width, height, 3, blurred_img, 100);

    //Liberación de espacio usado por el kernel
    for(size_t i = 0; i < kernel_size; ++i) 
        free(kernel[i]);
    
    free(kernel);

    //Liberación de espacio usado para codificación de la imágen
    stbi_image_free(img);

    //Liberación de espacio usado para codificación de imágen con filtro
    free(blurred_img);

    //Tiempo total(Elapsed Wall time)
    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);

    double seconds_d = (double)micros / pow(10,6);  

    //Devolver tiempo de ejecución del programa
    printf("\nTiempo de ejecucion: %f segundos\n", seconds_d);

    FILE * fp;
    int i;  
    size_t file_length = strlen(argv[1]) + strlen(argv[3]) + 6;
    char * fileName = (char *)malloc(sizeof(char)*file_length);
    fileName[0] = '\0';
    strcat(fileName, argv[1]);
    strcat(fileName, "_");
    strcat(fileName, argv[3]);
    strcat(fileName, ".txt");

    //Abrir archivo para registrar el tiempo medido
    fp = fopen (fileName,"a");
    
    fprintf (fp, "%f ", seconds_d);
   
    fclose (fp);
   
    return EXIT_SUCCESS;
}