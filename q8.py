class GFG:
    def dfs(self, curr, des, adj, vis):
        if curr == des:
            return True
        vis[curr] = 1
        for x in adj[curr]:
            if not vis[x]:
                if self.dfs(x, des, adj, vis):
                    return True
        return False

    def isPath(self, src, des, adj):
        vis = [0] * (len(adj) + 1)
        return self.dfs(src, des, adj, vis)

    def findSCC(self, n, a):
        ans = []
        is_scc = [0] * (n + 1)
        adj = [[] for _ in range(n + 1)]

        for i in range(len(a)):
            adj[a[i][0]].append(a[i][1])

        for i in range(1, n + 1):
            if not is_scc[i]:
                scc = [i]
                for j in range(i + 1, n + 1):
                    if not is_scc[j] and self.isPath(i, j, adj) and self.isPath(j, i, adj):
                        is_scc[j] = 1
                        scc.append(j)
                ans.append(scc)
        return ans

if __name__ == "__main__":
    obj = GFG()

    # Mapping family names to numeric values
    family_map = {"f1": 1, "f2": 2, "f3": 3, "f4": 4, "f5": 5, "f6": 6, "f7": 7, "f8": 8}

    V = 8  # Total number of families (nodes)
    edges = [
        [family_map["f1"], family_map["f5"]],
        [family_map["f2"], family_map["f1"]],
        [family_map["f2"], family_map["f3"]],
        [family_map["f2"], family_map["f7"]],
        [family_map["f3"], family_map["f5"]],
        [family_map["f3"], family_map["f6"]],
        [family_map["f5"], family_map["f2"]],
        [family_map["f6"], family_map["f3"]],
        [family_map["f6"], family_map["f4"]],
        [family_map["f6"], family_map["f8"]],
        [family_map["f7"], family_map["f4"]],
        [family_map["f8"], family_map["f1"]],
        [family_map["f8"], family_map["f7"]]
    ]

    ans = obj.findSCC(V, edges)

    # Reverse mapping to display original family names
    reverse_map = {v: k for k, v in family_map.items()}

    print("Strongly Connected Components are:")
    for x in ans:
        for y in x:
            print(reverse_map[y], end=" ")
        print()