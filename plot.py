def do_component_blondin_plot(self, axis_1=0, axis_2=1, marker_size=40):
    indicators = self.spectral_indicators

    s1 = indicators["EWSiII6355"]
    s2 = indicators["EWSiII5972"]

    plt.figure()

    cut = s2 > 30
    plt.scatter(
        self.embedding[cut, axis_1],
        self.embedding[cut, axis_2],
        s=marker_size,
        c="r",
        label="Cool (CL)",
    )
    cut = (s2 < 30) & (s1 < 70)
    plt.scatter(
        self.embedding[cut, axis_1],
        self.embedding[cut, axis_2],
        s=marker_size,
        c="g",
        label="Shallow silicon (SS)",
    )
    cut = (s2 < 30) & (s1 > 70) & (s1 < 100)
    plt.scatter(
        self.embedding[cut, axis_1],
        self.embedding[cut, axis_2],
        s=marker_size,
        c="black",
        label="Core normal (CN)",
    )
    cut = (s2 < 30) & (s1 > 100)
    plt.scatter(
        self.embedding[cut, axis_1],
        self.embedding[cut, axis_2],
        s=marker_size,
        c="b",
        label="Broad line (BL)",
    )

    plt.xlabel("Component %d" % (axis_1 + 1))
    plt.ylabel("Component %d" % (axis_2 + 1))

    plt.legend()

def do_blondin_plot(self, marker_size=40):
    indicators = self.spectral_indicators

    s1 = indicators["EWSiII6355"]
    s2 = indicators["EWSiII5972"]

    plt.figure()

    cut = s2 > 30
    plt.scatter(s1[cut], s2[cut], s=marker_size, c="r", label="Cool (CL)")
    cut = (s2 < 30) & (s1 < 70)
    plt.scatter(
        s1[cut], s2[cut], s=marker_size, c="g", label="Shallow silicon (SS)"
    )
    cut = (s2 < 30) & (s1 > 70) & (s1 < 100)
    plt.scatter(
        s1[cut], s2[cut], s=marker_size, c="black", label="Core normal (CN)"
    )
    cut = (s2 < 30) & (s1 > 100)
    plt.scatter(s1[cut], s2[cut], s=marker_size, c="b", label="Broad line (BL)")

    plt.xlabel("SiII 6355 Equivalent Width")
    plt.ylabel("SiII 5972 Equivalent Width")

    plt.legend()

def do_blondin_plot_3d(self, marker_size=40):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=Axes3D.name)

    indicators = self.spectral_indicators

    s1 = indicators["EWSiII6355"]
    s2 = indicators["EWSiII5972"]

    embedding = self.embedding

    cut = s2 > 30
    ax.scatter(
        embedding[cut, 0],
        embedding[cut, 1],
        embedding[cut, 2],
        s=marker_size,
        c="r",
        label="Cool (CL)",
    )
    cut = (s2 < 30) & (s1 < 70)
    ax.scatter(
        embedding[cut, 0],
        embedding[cut, 1],
        embedding[cut, 2],
        s=marker_size,
        c="g",
        label="Shallow silicon (SS)",
    )
    cut = (s2 < 30) & (s1 > 70) & (s1 < 100)
    ax.scatter(
        embedding[cut, 0],
        embedding[cut, 1],
        embedding[cut, 2],
        s=marker_size,
        c="black",
        label="Core normal (CN)",
    )
    cut = (s2 < 30) & (s1 > 100)
    ax.scatter(
        embedding[cut, 0],
        embedding[cut, 1],
        embedding[cut, 2],
        s=marker_size,
        c="b",
        label="Broad line (BL)",
    )

    ax.set_xlabel("Component 0")
    ax.set_ylabel("Component 1")
    ax.set_zlabel("Component 2")

    ax.legend()

