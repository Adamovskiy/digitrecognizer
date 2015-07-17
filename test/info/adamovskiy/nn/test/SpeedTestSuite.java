package info.adamovskiy.nn.test;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

// TODO seems like absence of warm up phase can cause enormous timing growing, whereas timing in real use cases is reduced.
/**
 * This class is not intended to be a good macro-benchmark tool (at least for
 * now). Only usage of result timings - rough estimate of the magnitude of
 * changes.
 *
 */
public abstract class SpeedTestSuite {
	@Retention(RetentionPolicy.RUNTIME)
	@Target({ElementType.METHOD})
	protected @interface SpeedTest {
		int order() default Integer.MAX_VALUE;
		String initMethod() default "";
	}
	
	private static class TimedMessage {
		public final String message;
		public final long timestamp;
		public TimedMessage(String message, long timestamp) {
			this.message = message;
			this.timestamp = timestamp;
		}
	}
	
	protected void launch() {
		this.initSuite();
		System.out.println("Speed testing suite " + this.getClass().getName() + " is started.");
		Method[] methods = this.getClass().getMethods();
		Arrays.sort(methods, new Comparator<Method>() {
			@Override
			public int compare(Method m1, Method m2) {
				final SpeedTest annotation1 = m1.getAnnotation(SpeedTest.class);
				final SpeedTest annotation2 = m2.getAnnotation(SpeedTest.class);
				if (annotation1 == null || annotation2 == null)
					return 0;
				return annotation1.order() - annotation2.order();
			}
		});
		for (Method method : methods) {
			SpeedTest annotation = method.getAnnotation(SpeedTest.class);
			if (annotation == null) {
				continue;
			}
			System.out.println();
			System.out.println("* Method: " + method.getName());
			if (!annotation.initMethod().isEmpty())
				try {
					this.getClass().getMethod(annotation.initMethod()).invoke(this);
				} catch (Exception e) {
					throw new IllegalStateException(String.format("Bad init method name \"%s\"for test \"%s\"", method.getName(), annotation.initMethod()), e);
				}
			long startTime = System.currentTimeMillis();
			try {
				method.invoke(this);
				this.addMessage(String.format("Method %s performed successfully.", method.getName()));
			} catch (Throwable e) {
				StringWriter sw = new StringWriter();
				e.printStackTrace(new PrintWriter(sw));
				this.addMessage(String.format("Method %s thrown %s.\n", method.getName(), e.getClass()
						.getName())
						+ sw.toString());
			}
			finally {
//				long lastTime = startTime;
				for (TimedMessage timedMessage : this.messages) {
					String printedTime = timedMessage.timestamp - startTime < 100 ? "<100" :
							Long.toString(timedMessage.timestamp - startTime);
					System.out.println(String.format("\t%s\t%s", printedTime, timedMessage.message));
//					if (lastTime - timedMessage.timestamp < 100)
//						System.out.println("\tWARNING: result timing can be invalid. Consider the macrobenchmark instrument!");
//					lastTime = timedMessage.timestamp;
				}
				messages.clear();
			}
		}
		System.out.println();
		System.out.println("Speed testing is finished.");
	}
	
	private List<TimedMessage> messages;
	
	public SpeedTestSuite() {
		messages = new ArrayList<TimedMessage>();
	}
	
	protected void addMessage(String message) {
		messages.add(new TimedMessage(message, System.currentTimeMillis()));
	}
	
	protected void initSuite() {
		// stub
	}
}
