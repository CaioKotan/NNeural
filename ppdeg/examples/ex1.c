/*

title: NNeural-Perceptron

auth: Caio A. Nog
date: 10/14/2026

file: ex1.c

*/
#include <stdio.h>
#include "../perceptron.h" //aqui você ajusta conforme o codigo do perceptron se encontra no seu projeto

int main(void){
    Perceptron p; //cria um perceptron "p"
    float inputs[] = {1.0f, 1.0f};//define inputs fora do perceptron

    if (!perceptron_init(&p, 2)) { //inicializa o perceptron criado
        printf("Erro ao inicializar perceptron.\n");
        return 1;
    }

    printf("Antes do treino: %d\n", //testa o perceptron antes do treino
           perceptron_predict(&p, inputs));

    for (int i = 0; i < 10; ++i) { //treina o perceptron 10 vezes usando os inputs definidos anteriormente
        perceptron_train(&p, inputs, 1, 0.1f); //taxa de aprendizado em 0.1f
    }

    printf("Depois do treino: %d\n", //testa o perceptron depois do treino
           perceptron_predict(&p, inputs));

    perceptron_free(&p); //libera a memoria armazenada

    return 0;
}