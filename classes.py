
# keeps track of increment
class Iterator:
    def __init__(self):
        self.max = float('inf')
        self.min = float('-inf')
        self.i = 0

    def set_max(self, maximum):
        self.max = maximum
        
    def set_min(self, minimum):
        self.min = minimum

    def increment(self):
        self.i = self.i+1 if self.i<self.max else self.min
        
    def decrement(self):
        self.i = self.i-1 if self.i>self.min else self.max
        
    def get(self):
        return self.i

    def set(self, value):
        
        value = value if value<=self.max else self.max
        value = value if value>=self.min else self.min
        self.i = value
        
    def set_mapping(self, array):
        
        self.map = array
        
    def get_mapping(self):
        try:
            return self.map[self.i]
        except:
            return None

# keeps track of region data
class Region:
    def __init__(self):
        self.parent = dict()
        
    def add(self,orbit, bound, value):
        
        if (orbit==None) or (bound==None):
            return
        
        try:
            self.parent[orbit][bound] = value
        except:
            self.parent[orbit] = dict()
            self.parent[orbit][bound] = value
            
    def get_value(self,orbit, bound):
        try:
            return self.parent[orbit][bound]
        except:
            return None
        
    def get_orbit(self,orbit):
        try:
            return self.parent[orbit]
        except:
            return None
        
    def get(self):
        return self.parent

    def get_orbit_keys(self,orbit):
        try:
            return list(self.parent[orbit])
        except:
            return None
        
    def get_keys(self):
        try:
            return list(self.parent)
        except:
            return None

    def get_orbit_values(self,orbit):
        try:
            keys = self.get_orbit_keys(orbit)

            return [self.parent[orbit][key] for key in keys]
        except:
            return None
        
    def get_orbit_values_pair(self,orbit):
        try:
            keys = self.get_orbit_keys(orbit)

            return [(self.parent[orbit][key], key) for key in keys]
        except:
            return None
        
    def del_entry(self,orbit,bound):
        try:
            del self.parent[orbit][bound]
        except:
            pass
