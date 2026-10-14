/*

title: NNeural-Perceptron

auth: Caio A. Nog
date: 10/14/2026

file: perceptron.h

*/
#ifndef PERCEPTRON_H //caso nao tenho definido, le adiante
#define PERCEPTRON_H

#include <stddef.h> //utilizei o size_t desse modulo
//com o size_t posso ter as entradas enquanto variaveis

typedef struct {
    size_t n_inputs; //numero de entradas
    float *weights; //pesos || o uso de "*" e para nao limitarmos a um numero especifico de pesos
    float bias; //valor somado ao calculo
} Perceptron;

int perceptron_init(Perceptron *p, size_t n_inputs); //inicia a struct Perceptron
int perceptron_predict(const Perceptron *p, const float *inputs); //executa os calculos do perceptron, por si so nao altera nenhum dado de entrada ou pesos
int perceptron_train(Perceptron *p, const float *inputs,
                     int target, float learning_rate); //treina o perceptron, com base em um alvo e uma taxa de aprendizado

void perceptron_free(Perceptron *p); //libera a memoria armazenada

#endif
