// Contorno para compilar CUDA sobre glibc 2.41 ou mais nova.
//
// O PROBLEMA
//
// A glibc 2.41 passou a declarar as funcoes de C23 cospi, sinpi e rsqrt em
// <math.h>, com __THROW - ou seja, noexcept(true). O crt/math_functions.h do
// CUDA declara as MESMAS funcoes sem noexcept, e o frontend do nvcc rejeita:
//
//   error: exception specification is incompatible with that of previous
//          function "cospi"
//
// Como o cuda_runtime.h e pre-incluido em todo .cu, isso derruba ate um
// arquivo com um kernel vazio e nenhum include. Nao e problema do codigo do
// projeto - e do par toolkit/glibc:
//
//   CUDA 12.9   nao guarda nenhuma das tres        -> falha
//   CUDA 13.1   guarda sinpi e cospi, esquece rsqrt -> falha
//
// A SOLUCAO
//
// Fazer a glibc nao declarar essas tres. Em bits/libc-header-start.h a guarda e
//
//   __GLIBC_USE (IEC_60559_FUNCS_EXT) || __GLIBC_USE (ISOC23)
//
// e as duas pontas precisam cair. A primeira sai com __USE_GNU desligado; a
// segunda NAO sai junto, porque _GNU_SOURCE tambem define _ISOC23_SOURCE, e
// features.h tem guarda de inclusao - desligar _GNU_SOURCE depois nao refaz a
// conta. Por isso __GLIBC_USE_ISOC23 e forcado a 0 na mao.
//
// Este arquivo entra na frente do <math.h> do sistema por -I, e o include_next
// pega o de verdade. Os macros sao restaurados depois, entao so a declaracao
// dessas funcoes muda - e nenhuma delas e usada aqui.
//
// Tentado e sem efeito, para nao repetir: -allow-unsupported-compiler,
// -U_GNU_SOURCE (o libstdc++ redefine _GNU_SOURCE), -ccbin com g++-13 ou
// g++-14 (o cabecalho em conflito e da glibc, nao do GCC), e -include de um
// cabecalho de correcao (chega depois do cuda_runtime.h pre-incluido).
//
// Em sistema com glibc anterior a 2.41 este arquivo nao muda nada: as funcoes
// nao sao declaradas la de qualquer forma.

#pragma once

#include <features.h>

#undef __USE_GNU
#undef __GLIBC_USE_ISOC23
#define __GLIBC_USE_ISOC23 0

#include_next <math.h>

#undef __GLIBC_USE_ISOC23
#define __GLIBC_USE_ISOC23 1
#define __USE_GNU 1
