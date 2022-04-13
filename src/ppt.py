import matplotlib.pyplot as plt
from scipy.stats import geom


def plot(p, scale, title, filename):
    with plt.style.context('dark_background'):
        X = [1, 2, 3, 4, 5, 6, 7]
        geom_pd = geom.pmf(X, p) * scale

        plt.figure()
        plt.ylabel("Normalized Frequency")
        plt.xlabel("Rank")
        plt.ylim(0, 1)
        plt.title(title)
        plt.vlines(X, 0, geom_pd, color='r', lw=20, alpha=0.5)
        plt.savefig(filename)
        plt.close()


if __name__ == '__main__':
    plot(0.5, 1.5, "Low Entropy Distribution", "low-ent.pdf")
    plot(0.1, 4, "High Entropy Distribution", "high-ent.pdf")

    with plt.style.context('dark_background'):
        x = [0, 1, 2, 3, 4, 5, 6]
        y = [0.2, 0.4, 0.3, 0.5, 0.6, 0.5, 0.7]

        plt.figure()
        plt.annotate("", xy=(x[-1], y[-1]), xytext=(x[-2], y[-2]),
                     arrowprops=dict(color='orange', headwidth=10,
                                     headlength=10, width=0.1))
        plt.plot(x, y, color='orange', alpha=1.0)
        plt.plot(x[:-1], y[:-1], 'o', color='orange')
        plt.ylabel("Normalized Entropy")
        plt.xlabel("Time Index")
        plt.ylim(0, 1)
        plt.title("Time Series")
        plt.savefig("ts.pdf")
        plt.close()
