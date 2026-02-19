
class Trie:

	def __init__(self):
		self.root = [dict(), None]
		self.size = 0
	
	def add(self, keys, value):
		current = self.root
		for key in keys:
			next = current[0].get(key)
			if next is None:
				next = [dict(), None]
				current[0][key] = next
			current = next
		old_value = current[1]
		current[1] = value
		if old_value is None and not value is None:
			self.size += 1
		return old_value
	
	def get(self, keys):
		current = self.root
		for key in keys:
			current = current[0].get(key)
			if current is None:
				return None
		return current[1]
	
	def lookup(self, keys):
		keys2 = list(keys)
		keys2.append(None)
		completed = list()
		current = list()
		for index, key in enumerate(keys2):
			#print("index = {} key = {}".format(index, key))
			next = list()
			current.append([index, self.root[0], self.root[1]])
			# Consider current spans; three possibilities: continue a span, end a span, complete a span
			for span_start, node_children, node_value in current:
				#print("Checking span_start = {} node_children = {} node_value = {}".format(span_start, node_children, node_value))
				if not node_value is None:
					completed.append((node_value, span_start, index))
				child = node_children.get(key)
				#print("Checking child[key] = {}".format(child))
				if not child is None:
					next.append([span_start, child[0], child[1]])
			current = next
		return completed
	
	def __len__(self):
		return self.size
