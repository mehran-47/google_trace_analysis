import sys
import os
import shutil

class Dir:
	def __init__(self, *args):
		self.dirPath = os.getcwd() if len(args)==0 else args[0]
		if not os.path.exists(self.dirPath):
			os.makedirs(self.dirPath)

	def deleteDir(self, *args):
		if len(args) == 0 :
			shutil.rmtree(self.dirPath)
		else:
			for aDir in args:
				if os.path.exists(aDir):
					shutil.rmtree(aDir)

	def moveIn(self, *args):
		if len(args) == 0 :
			return
		else:
			for srcPath in args:
				if os.path.exists(srcPath):
					shutil.move(srcPath, self.dirPath)

	def getDir(self):
		return self.dirPath

"""
DO = Dir('/home/mk/Downloads/immaDO')
DO.moveIn('/home/mk/Downloads/testDir', '/home/mk/Downloads/Heat-workshop-June14-mt-ph.ppt')
DO.deleteDir()
"""