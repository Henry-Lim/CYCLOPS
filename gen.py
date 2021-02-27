import pandas as pd
import numpy as np

class Gene:
    def __init__(self, w, phi, amplitude=1.0, mean=0.0, noise_sd=0.15, name=None):
        self.w = w
        self.phi = phi
        self.amplitude = amplitude
        self.mean = mean
        self.noise_sd = noise_sd
        self.name = name
    
    def expression(self, t, noisy=True):
        if self.w == 0:
            expr = np.random.normal(scale=0.1)
        else:
            expr = self.mean + self.amplitude*np.cos(self.w*t + self.phi)

        if noisy:
            expr += np.random.normal(scale=self.noise_sd)
        return expr
    
    
class Cell:
    def __init__(self, t, name=None, genes=[]):
        self.t = t
        self.genes = genes
        self.expressions = self.express_genes()
        self.name = name
        
    def express_genes(self, noisy=True):
        gene_names = []
        gene_expressions = []
        for gene in self.genes:
            gene_names.append(gene.name)
            gene_expressions.append(gene.expression(self.t, noisy))
        
        self.expressions = pd.Series(gene_expressions, index=gene_names)
    
        return self.expressions