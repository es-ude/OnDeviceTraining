/* Comments may mention qc->scale, qc->scales[g] = x and *dstQC = *srcQC. */
float f(symInt32QConfig_t *qc, symQConfig_t *sq, const symQConfig_t *src, size_t g) {
    // a line comment about ->scale
    const char *msg = "qc->scale must be > 0";
    (void)msg;
    storeCanonScale(&qc->scale, 0.5f);
    odtScaleStore(&sq->scales[g], ODT_SCALE_ONE);
    copyCanonScales(sq->scales, src->scales, src->numGroups);
    const float s = loadCanonScale(&((symInt32QConfig_t *)qc)->scale);
    return s * odtScaleToFloat(odtScaleLoad(&sq->scales[0]));
}
