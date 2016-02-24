import numpy as np

def main():
    mat = np.loadtxt(open("actives_matrix.csv","rb"),delimiter=",",skiprows=0)
    
    print mat.max(axis=0)
    max_points = [.03 * maximum for maximum in mat.max(axis=0)]

    print(np.median(max_points))

if __name__=="__main__":
    main()