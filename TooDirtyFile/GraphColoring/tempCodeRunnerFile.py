    if (Is_broken == "D"):
                        if text_bits2[i][k][j] is None: 
                            text_bits2[i][k][j] = ax[1].text(1+k, 1+i, "  ", ha='center', va='center', fontsize=5,bbox=dict(facecolor="green", edgecolor='white', boxstyle='round', pad=0.5))
                        else:
                            text_bits2[i][k][j].set_text("  ")
                            text_bits2[i][k][j].set_bbox(dict(facecolor="green", edgecolor='white', boxstyle='round', pad=0.5))    

                    elif (Is_broken == "N"):
                        if text_bits2[i][k][j] is None: 
                            text_bits2[i][k][j] = ax[1].text(1+k, 1+i, "  ", ha='center', va='center', fontsize=5,bbox=dict(facecolor="red", edgecolor='white', boxstyle='round', pad=0.5))
                        else:
                            text_bits2[i][k][j].set_text("  ")
                            text_bits2[i][k][j].set_bbox(dict(facecolor="red", edgecolor='white', boxstyle='round', pad=0.5))    

                    elif (Is_broken == "O"):
                        if text_bits2[i][k][j] is not None:
                            text_bits2[i][k][j].set_text("")
                            text_bits2[i][k][j].set_bbox(dict(facecolor="white", edgecolor='white', boxstyle='round', pad=0))
                    
                    if array_new[cnt][i*w+k][j] == 1:
                        if text_bits1[i][k][j] is None:
                            text_bits1[i][k][j] = ax[0].text(1+k, 1+i, j, ha='center', va='center', fontsize=5,bbox=dict(facecolor=color[j], edgecolor='white', boxstyle='round', pad=0.5))
                        else:
                            text_bits1[i][k][j].set_text(j)
                            text_bits1[i][k][j].set_bbox(dict(facecolor=color[j], edgecolor='white', boxstyle='round', pad=0.5))
                    else:
                        if text_bits1[i][k][j] is not None:
                            text_bits1[i][k][j].set_text("")
                            text_bits1[i][k][j].set_bbox(dict(facecolor="white", edgecolor='white', boxstyle='round', pad=0))

