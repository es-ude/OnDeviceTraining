void f(symInt32QConfig_t *qc) {
    float *bits = (float *)&qc->scale;
    *bits = 0.5f;
}
