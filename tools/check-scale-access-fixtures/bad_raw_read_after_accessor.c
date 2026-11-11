float f(const symInt32QConfig_t *a, const symInt32QConfig_t *b) {
    /* raw read after an accessor in the same expression: the accessor has
     * already closed, so b->scale is not its argument */
    return loadCanonScale(&a->scale) * odtScaleToFloat(b->scale);
}
