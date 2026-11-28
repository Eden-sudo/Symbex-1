#pragma once
#ifdef __cplusplus
extern "C" {
#endif
int l2c_init(const void *model, int cap);   /* returns the context allocated */
long l2c_state_bytes(const void *model, int cap);  /* bytes run.c's malloc_run_state asks for (l2c_init's callocs) */
int l2c_step(int token, int pos);           /* forward one token, returns the greedy next token */
float l2c_logit(int i);                     /* the last step's logit i */
#ifdef __cplusplus
}
#endif
