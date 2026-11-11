void f(quantization_t *dst, const quantization_t *src) {
    memcpy(dst->qConfig, src->qConfig, sizeof(symInt32QConfig_t));
}
