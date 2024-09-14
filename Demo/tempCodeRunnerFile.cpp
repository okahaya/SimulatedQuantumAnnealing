    for (int i = start; i < end; ++i) {
        Q[i][i] = 1 - 2 * n;
        for (int j = i + 1; j < end; ++j) {
            Q[i][j] = 2;
            Q[j][i] = 2;
        }
    }