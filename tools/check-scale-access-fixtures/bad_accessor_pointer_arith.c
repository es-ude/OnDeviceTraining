float f(symQConfig_t *qc) {
    return loadCanonScale(&qc->scales[0] + 1);
}
